import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import numpy as np
from pulp import * 


def run_pulp_solver(data):
    """
    Construye y resuelve el modelo de PLEM para el problema de transporte.
    Recibe un diccionario 'data' con todos los parámetros de la red.
    """
    
    "A partir de data construimos las variables y vectores necesarios "
    I = range(data['M'])  
    J = range(data['N'])  
    K = range(data['K'])
    
    v_params = data['vehiculos']
    Cap_C = v_params['capacidad_camion']
    CF_C = v_params['coste_fijo_camion']
    Cap_F = v_params['capacidad_furgoneta']
    CF_F = v_params['coste_fijo_furgoneta']
    
    Oferta = data['oferta_O']
    Demanda = data['demanda_D']
    Coste_O_A = data['matriz_O_A']
    Coste_A_D = data['matriz_A_D']
    Coste_Fijo_A = data['costes_fijos_A']
    Cap_A = data['capacidad_A']

    "Empezamoz a construir el modelo de optimización"
    modelo = LpProblem("TransporteCrumblEuropa", LpMinimize)

    
    "Definimos variables de decisión"
    # Flujo de Mercancía (x: O->A, y: A->D)
    x_ij = LpVariable.dicts("Flujo_O_A", [(i, j) for i in I for j in J], lowBound=0, cat='Continuous') 
    y_jk = LpVariable.dicts("Flujo_A_D", [(j, k) for j in J for k in K], lowBound=0, cat='Continuous')

    # Uso de Instalaciones (w: Apertura de Almacén)
    w_j = LpVariable.dicts("Abrir_Almacen", J, cat='Binary') 

    # Uso de Vehículos (v: Camiones O->A, u: Furgonetas A->D)
    v_ij = LpVariable.dicts("Num_Camiones_O_A", [(i, j) for i in I for j in J], lowBound=0, cat='Integer')
    u_jk = LpVariable.dicts("Num_Furgonetas_A_D", [(j, k) for j in J for k in K], lowBound=0, cat='Integer')

    
    # Costes Fijos (Almacenes + Vehículos)
    CF_Total = sum(Coste_Fijo_A[j] * w_j[j] for j in J) + \
               sum(CF_C * v_ij[i, j] for i in I for j in J) + \
               sum(CF_F * u_jk[j, k] for j in J for k in K)
    
    # Costes Variables
    CV_Total = sum(Coste_O_A[i, j] * x_ij[i, j] for i in I for j in J) + \
               sum(Coste_A_D[j, k] * y_jk[j, k] for j in J for k in K)
    
    modelo += CF_Total + CV_Total, "Costo_Total"

    
    "Restricciones"
    # R1: Restricción de Demanda
    for k in K:
        modelo += sum(y_jk[j, k] for j in J) >= Demanda[k], f"R_Demanda_{k}"

    # R2: Restricción de Oferta
    for i in I:
        modelo += sum(x_ij[i, j] for j in J) <= Oferta[i], f"R_Oferta_{i}"

    # R3: Restricción de Balance/Transbordo
    for j in J:
        modelo += sum(x_ij[i, j] for i in I) == sum(y_jk[j, k] for k in K), f"R_Transbordo_{j}"

    # R4: Capacidad del Vehículo (Flujo <= Num_Vehículos * Capacidad)
    for i in I:
        for j in J:
            modelo += x_ij[i, j] <= v_ij[i, j] * Cap_C, f"R_Cap_Camion_{i}_{j}"

    for j in J:
        for k in K:
            modelo += y_jk[j, k] <= u_jk[j, k] * Cap_F, f"R_Cap_Furgoneta_{j}_{k}"
            
    # R5: Restricción de Capacidad y Apertura del Almacén (Big-M)
    for j in J:
        modelo += sum(x_ij[i, j] for i in I) <= Cap_A[j] * w_j[j], f"R_Cap_Max_Almacen_{j}"

    "Resolvemos"
    try:
        modelo.solve()
    except Exception as e:
        return {"estado": "ERROR_SOLVER", "mensaje": str(e)}

    return modelo


class SimpleSupplyChainApp:
    """
    Inicializar las variables de la
    aplicación de interfaz gráfica (GUI) de Tkinter
    """

    default_matriz_O_A = None
    default_matriz_A_D = None

    def __init__(self, master):
        self.master = master
        master.title("Optimizador de Logística Crumbl (PuLP + Tkinter)")
        
        self.M = 0
        self.N = 0
        self.K = 0
        self.data = {}
        self.entries_O_A = []
        self.entries_A_D = []
        self.resultado_texto = None 
        
        if SimpleSupplyChainApp.default_matriz_O_A is None:
             # Valores por defecto para la primera ejecución (2x3)
            SimpleSupplyChainApp.default_matriz_O_A = np.array([[10, 15, 20], [25, 12, 18]]) 
            SimpleSupplyChainApp.default_matriz_A_D = np.array([[5, 8, 10, 12], [7, 6, 9, 11], [15, 14, 13, 16]])
        
        self.setup_initial_ui()


    def _create_label_entry(self, parent, label_text, row, default_value):
        
        """Auxiliar para crear Label y Entry."""
        
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky='w', pady=5)
        entry = ttk.Entry(parent, width=10)
        entry.grid(row=row, column=1, pady=5, padx=5)
        entry.insert(0, default_value)
        return entry
    
    
    def _create_vector_input(self, parent_frame, count, label_base, unit, col_offset=0, default_values=None):
        """
        Generar una columna de campos de entrada
        """
        
        entries = []

        if default_values is None or len(default_values) != count:
            default_values = ["" for _ in range(count)]
            
        for i in range(count):
            ttk.Label(parent_frame, text=f"{label_base} {i+1} ({unit}):").grid(
                row=i + 1,
                column=0 + col_offset,
                sticky='w',
                padx=5,
                pady=2
            )
            
            entry = ttk.Entry(parent_frame, width=15)
            entry.grid(
                row=i + 1,
                column=1 + col_offset,
                sticky='we',
                padx=5,
                pady=2
            )
            
            entry.insert(0, str(default_values[i])) 
            
            entries.append(entry)
        return entries
    
            
    def _create_matrix_input(self, parent_frame, rows, cols, row_label, col_label, default_matrix):
        """Auxiliar para crear una rejilla de entrada para matrices, usando una matriz de valores predeterminados."""
        
        if default_matrix is None or default_matrix.shape != (rows, cols):
            new_matrix = np.zeros((rows, cols))
            for i in range(rows):
                for j in range(cols):
                    new_matrix[i, j] = 5 + (i * 2) + (j * 3) 
            default_matrix = new_matrix

        for j in range(cols):
            ttk.Label(parent_frame, text=f"{col_label} {j+1}", font=("Arial", 9, "bold")).grid(row=1, column=j+1, padx=5, pady=2)
        
        entries = []
        for i in range(rows):
            row_entries = []
            ttk.Label(parent_frame, text=f"{row_label} {i+1}", font=("Arial", 9, "bold")).grid(row=i+2, column=0, padx=5, pady=2)
            
            for j in range(cols):
                entry = ttk.Entry(parent_frame, width=8)
                entry.grid(row=i+2, column=j+1, padx=5, pady=2)
                
                entry.insert(0, str(default_matrix[i, j])) 
                
                row_entries.append(entry)
            entries.append(row_entries)
            
        if row_label == "Origen":
            SimpleSupplyChainApp.default_matriz_O_A = default_matrix
        elif row_label == "Almacén":
            SimpleSupplyChainApp.default_matriz_A_D = default_matrix
            
        return entries
        
    def _create_single_input(self, parent_frame, row, col_offset, label_text, default_value):
        """Auxiliar para crear un solo campo de entrada."""
        
        ttk.Label(parent_frame, text=label_text).grid(row=row, column=0 + col_offset, sticky='w', padx=5, pady=2)
        entry = ttk.Entry(parent_frame, width=15)
        entry.grid(row=row, column=1 + col_offset, padx=5, pady=2)
        entry.insert(0, default_value)
        return entry

    def _create_vehicle_inputs(self, parent_frame):
        """Crea los campos de entrada para los parámetros de vehículos."""
        
        ttk.Label(parent_frame, text="Configuración de la Flota de Transporte", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, pady=10)
        
        self.entry_camion_coste = self._create_single_input(parent_frame, 1, 0, "Costo Fijo Camión (€):", "500.0")
        self.entry_camion_capacidad = self._create_single_input(parent_frame, 2, 0, "Capacidad Camión (unidades):", "1000")

        self.entry_furgoneta_coste = self._create_single_input(parent_frame, 3, 0, "Costo Fijo Furgoneta (€):", "150.0")
        self.entry_furgoneta_capacidad = self._create_single_input(parent_frame, 4, 0, "Capacidad Furgoneta (unidades):", "300")
        

    def setup_initial_ui(self):
        """Paso 1: Pide M, N y K iniciales."""
        
        for widget in self.master.winfo_children():
            widget.destroy()

        frame = ttk.Frame(self.master, padding="15")
        frame.pack(padx=10, pady=10)

        ttk.Label(frame, text="Paso 1: Definir Dimensiones", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=10)

        self.entry_M = self._create_label_entry(frame, "Orígenes (M):", 1, "2")
        self.entry_N = self._create_label_entry(frame, "Almacenes (N):", 2, "3")
        self.entry_K = self._create_label_entry(frame, "Destinos (K):", 3, "4")
        
        ttk.Button(frame, text="Continuar (Paso 2)", command=self.process_dimensions).grid(row=4, column=0, columnspan=2, pady=15)
        
    def process_dimensions(self):
        """Valida M, N, K, actualiza las matrices por defecto si las dimensiones cambian, y pasa a la recolección de datos."""
        
        try:
            M_new = int(self.entry_M.get())
            N_new = int(self.entry_N.get())
            K_new = int(self.entry_K.get())
            
            if M_new <= 0 or N_new <= 0 or K_new <= 0:
                raise ValueError
            
            self.M, self.N, self.K = M_new, N_new, K_new

            if SimpleSupplyChainApp.default_matriz_O_A is None or SimpleSupplyChainApp.default_matriz_O_A.shape != (self.M, self.N):
                new_matrix_O_A = np.zeros((self.M, self.N))
                for i in range(self.M):
                    for j in range(self.N):
                        new_matrix_O_A[i, j] = 10 + (i * 5) + (j * 3) # Valor predeterminado fijo
                SimpleSupplyChainApp.default_matriz_O_A = new_matrix_O_A

            if SimpleSupplyChainApp.default_matriz_A_D is None or SimpleSupplyChainApp.default_matriz_A_D.shape != (self.N, self.K):
                new_matrix_A_D = np.zeros((self.N, self.K))
                for i in range(self.N):
                    for j in range(self.K):
                        new_matrix_A_D[i, j] = 5 + (i * 2) + (j * 4) # Valor predeterminado fijo
                SimpleSupplyChainApp.default_matriz_A_D = new_matrix_A_D

            self.collect_all_vectors()
            
        except ValueError:
            messagebox.showerror("Error de Entrada", "M, N y K deben ser números enteros y positivos.")


    def collect_all_vectors(self):
        """Paso 2: Pide todos los vectores (Oferta, Capacidad, Costes Fijos, Demanda) con valores por defecto."""
        
        # Oferta (M)
        default_oferta = [5000 + i * 1000 for i in range(self.M)]
        
        # Almacenes (N)
        default_capacidad_A = [10000 + j * 500 for j in range(self.N)] # Fijo y editable
        default_costes_fijos_A = [50000.0 + j * 10000 for j in range(self.N)] # Fijo y editable
        
        # Demanda (K)
        demanda_unit = sum(default_oferta) * 0.6 / self.K 
        default_demanda = [int(demanda_unit) for _ in range(self.K)] # Fijo y editable
        
        for widget in self.master.winfo_children():
            widget.destroy()
        
        main_frame = ttk.Frame(self.master, padding="15")
        main_frame.pack(fill='both', expand=True)
        
        ttk.Label(main_frame, text="Paso 2: Definir Oferta, Capacidad y Demanda", font=("Arial", 14, "bold")).pack(pady=10)
        
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        frame_oferta = ttk.Frame(notebook, padding="10")
        notebook.add(frame_oferta, text=f'Oferta (M={self.M})')
        self.entry_oferta = self._create_vector_input(frame_oferta, self.M, "Oferta Origen", "unidades", default_values=default_oferta)
        
        frame_almacen = ttk.Frame(notebook, padding="10")
        notebook.add(frame_almacen, text=f'Almacenes (N={self.N})')
        
        frame_almacen.columnconfigure(0, weight=1)
        frame_almacen.columnconfigure(1, weight=1)
        frame_almacen.columnconfigure(2, weight=1)
        frame_almacen.columnconfigure(3, weight=1)
        
        ttk.Label(frame_almacen, text="Capacidad Máxima (unidades)", font=("Arial", 10, "bold")).grid(
            row=0, column=0, columnspan=2, padx=10, pady=5, sticky='w'
        )

        self.entry_capacidad_A = self._create_vector_input(frame_almacen, self.N, "Capacidad Almacén", "unidades", col_offset=0, default_values=default_capacidad_A)
        
        ttk.Label(frame_almacen, text="Costo Fijo Apertura (€)", font=("Arial", 10, "bold")).grid(
            row=0, column=2, columnspan=2, padx=10, pady=5, sticky='w'
        )

        self.entry_costes_fijos_A = self._create_vector_input(frame_almacen, self.N, "Costo Fijo Almacén", "€", col_offset=2, default_values=default_costes_fijos_A)
        

        frame_demanda = ttk.Frame(notebook, padding="10")
        notebook.add(frame_demanda, text=f'Demanda (K={self.K})')

        self.entry_demanda = self._create_vector_input(frame_demanda, self.K, "Demanda Destino", "unidades", default_values=default_demanda)
        

        ttk.Button(main_frame, text="Continuar (Paso 3: Matrices)", command=self.process_vectors_and_go_to_matrices).pack(pady=15)
        
    def process_vectors_and_go_to_matrices(self):
        """Procesa y guarda los vectores, y pasa a la recolección de matrices."""
        
        try:

            self.data['oferta_O'] = np.array([float(e.get()) for e in self.entry_oferta])
            self.data['capacidad_A'] = np.array([float(e.get()) for e in self.entry_capacidad_A])
            self.data['costes_fijos_A'] = np.array([float(e.get()) for e in self.entry_costes_fijos_A])
            self.data['demanda_D'] = np.array([float(e.get()) for e in self.entry_demanda])
            

            if np.sum(self.data['oferta_O']) < np.sum(self.data['demanda_D']):
                messagebox.showwarning("Advertencia de Viabilidad", "La Oferta Total es menor que la Demanda Total. El problema podría ser inviable.")

            self.collect_cost_matrices()

        except ValueError:
            messagebox.showerror("Error de Datos", "Asegúrese de que todos los valores numéricos son válidos (enteros o decimales).")


    def collect_cost_matrices(self):
        """Paso 3: Pide las matrices de costos O->A y A->D, y parámetros de vehículos."""
        
        for widget in self.master.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self.master, padding="15")
        main_frame.pack(fill='both', expand=True)

        ttk.Label(main_frame, text="Paso 3: Costos de Transporte y Flota 🚚🚐", font=("Arial", 14, "bold")).pack(pady=10)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True, padx=5, pady=5)

        frame_O_A = ttk.Frame(notebook, padding="10")
        notebook.add(frame_O_A, text=f'Costos O->A ({self.M}x{self.N})')
        
        ttk.Label(frame_O_A, text="Costo de transporte por unidad (€) de Origen a Almacén:", font=("Arial", 10)).grid(row=0, column=0, columnspan=self.N + 1, pady=5)
        
        self.entries_O_A = self._create_matrix_input(frame_O_A, self.M, self.N, "Origen", "Almacén", SimpleSupplyChainApp.default_matriz_O_A)

        frame_A_D = ttk.Frame(notebook, padding="10")
        notebook.add(frame_A_D, text=f'Costos A->D ({self.N}x{self.K})')
        
        ttk.Label(frame_A_D, text="Costo de transporte por unidad (€) de Almacén a Destino:", font=("Arial", 10)).grid(row=0, column=0, columnspan=self.K + 1, pady=5)
        
        self.entries_A_D = self._create_matrix_input(frame_A_D, self.N, self.K, "Almacén", "Destino", SimpleSupplyChainApp.default_matriz_A_D)
        
        frame_vehiculos = ttk.Frame(notebook, padding="10")
        notebook.add(frame_vehiculos, text='Flota de Vehículos')
        
        self._create_vehicle_inputs(frame_vehiculos)

        ttk.Button(main_frame, text="Resolver Optimización (Paso Final)", command=self.process_all_data).pack(pady=15)
    
    
    def process_all_data(self):
        """Recoge los datos finales de las matrices y vehículos y procede a resolver."""
        
        try:

            matriz_O_A_list = [[float(e.get()) for e in row_entries] for row_entries in self.entries_O_A]
            self.data['matriz_O_A'] = np.array(matriz_O_A_list)


            matriz_A_D_list = [[float(e.get()) for e in row_entries] for row_entries in self.entries_A_D]
            self.data['matriz_A_D'] = np.array(matriz_A_D_list)


            self.data['vehiculos'] = {
                'coste_fijo_camion': float(self.entry_camion_coste.get()),
                'capacidad_camion': int(self.entry_camion_capacidad.get()),
                'coste_fijo_furgoneta': float(self.entry_furgoneta_coste.get()),
                'capacidad_furgoneta': int(self.entry_furgoneta_capacidad.get())
            }
            

            self.data['M'] = self.M
            self.data['N'] = self.N
            self.data['K'] = self.K

            self.solve_optimization()

        except ValueError:
            messagebox.showerror("Error de Datos", "Asegúrese de que todos los valores de las matrices y vehículos son números válidos.")
         
            
    def solve_optimization(self):
        """Ejecuta el solver PuLP y muestra los resultados."""
        
        modelo = run_pulp_solver(self.data)
        
        for widget in self.master.winfo_children():
            widget.destroy()

        summary_frame = ttk.Frame(self.master, padding="15")
        summary_frame.pack(padx=10, pady=10, fill='both', expand=True)
        
        ttk.Label(summary_frame, text="🍪 😋 CRUMBL COOKIES 🥛 🍪", font=("Arial", 20, "bold")).pack(pady=15)

        
        ttk.Label(summary_frame, text="✅ SOLUCIÓN DE OPTIMIZACIÓN", font=("Arial", 16, "bold")).pack(pady=15)


        self.resultado_texto = tk.Text(summary_frame, height=30, width=80, wrap='word')
        self.resultado_texto.pack(padx=10, pady=10, fill='both', expand=True)
        

        if isinstance(modelo, dict) and modelo['estado'] == 'ERROR_SOLVER':
            self.resultado_texto.insert(tk.END, f"ERROR CRÍTICO: {modelo['mensaje']}\n\nAsegúrese de que PuLP está instalado correctamente (pip install pulp) y que tiene un solver disponible (como CBC o GLPK).")
        
        elif LpStatus[modelo.status] == 'Optimal':
            self._display_optimal_solution(modelo)
        else:
             self.resultado_texto.insert(tk.END, f"ESTADO: {LpStatus[modelo.status]}\n\nNo se encontró una solución óptima. El problema puede ser inviable o no tener solución finita.")
        
        self.resultado_texto.config(state=tk.DISABLED)

    def _display_optimal_solution(self, modelo):
        """Formatea y muestra la solución óptima en el widget de texto."""
        
        output = ""
        I, J, K = range(self.M), range(self.N), range(self.K)

        output += f"COSTO TOTAL MÍNIMO: €{value(modelo.objective):,.2f}\n"
        output += "-" * 50 + "\n"
        

        almacenes_abiertos = [j+1 for j in J if value(modelo.variablesDict()[f'Abrir_Almacen_{j}']) > 0.5]
        output += f"📍 ALMACENES HABILITADOS (N={self.N}): {', '.join([f'Almacén {a}' for a in almacenes_abiertos])}\n"
        output += "-" * 50 + "\n"
        

        output += "\n🚚 FLUJOS ORÍGENES -> ALMACENES (Camiones)\n"
        total_camiones = 0
        for i in I:
            for j in J:
                flujo = value(modelo.variablesDict()[f'Flujo_O_A_({i},_{j})'])
                num_veh = value(modelo.variablesDict()[f'Num_Camiones_O_A_({i},_{j})'])
                if flujo > 0.0001:
                    output += f"  O{i+1} -> A{j+1}: {flujo:,.0f} uds. | {num_veh:.0f} camiones usados (Coste Fijo: {num_veh * self.data['vehiculos']['coste_fijo_camion']:,.2f}€)\n"
                    total_camiones += num_veh
        output += f"TOTAL CAMIONES USADOS: {total_camiones:.0f}\n"


        output += "\n🚐 FLUJOS ALMACENES -> DESTINOS (Furgonetas)\n"
        total_furgonetas = 0
        for j in J:
            for k in K:
                flujo = value(modelo.variablesDict()[f'Flujo_A_D_({j},_{k})'])
                num_veh = value(modelo.variablesDict()[f'Num_Furgonetas_A_D_({j},_{k})'])
                if flujo > 0.0001:
                    output += f"  A{j+1} -> D{k+1}: {flujo:,.0f} uds. | {num_veh:.0f} furgonetas usadas (Coste Fijo: {num_veh * self.data['vehiculos']['coste_fijo_furgoneta']:,.2f}€)\n"
                    total_furgonetas += num_veh
        output += f"TOTAL FURGONETAS USADAS: {total_furgonetas:.0f}\n"
        
        self.resultado_texto.insert(tk.END, output)


"Punto de entrada para ejecutar Tkinter"
if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleSupplyChainApp(root)
    root.mainloop()