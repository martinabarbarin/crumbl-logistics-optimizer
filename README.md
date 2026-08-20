# Optimización de la Cadena de Suministro: Expansión de Crumbl Cookies en Europa

Diseño y desarrollo de un modelo de Programación Lineal Entera Mixta (PLEM) con interfaz gráfica para minimizar los costes logísticos en una red de transporte con transbordos.

## El Problema de Negocio
Como parte de una hipotética expansión de la marca **Crumbl Cookies** en Europa, este proyecto resuelve el problema de diseñar una red de distribución eficiente. La mercancía debe viajar desde las sedes de producción (Orígenes) hasta las tiendas (Destinos), pasando obligatoriamente por centros de clasificación (Almacenes Intermedios). 

El objetivo es **minimizar el coste total**, decidiendo:
* Qué almacenes intermedios resulta rentable abrir.
* Cuántos camiones y furgonetas utilizar.
* Qué cantidad de mercancía enviar por cada ruta disponible.

## Stack Técnico y Metodología
* **Lenguaje:** Python
* **Motor de Optimización:** `PuLP` (Modelado matemático y resolución del problema de Programación Lineal).
* **Cálculo Numérico:** `NumPy`.
* **Interfaz Gráfica (GUI):** `Tkinter` (Desarrollo de ventanas interactivas para que usuarios sin conocimientos de código puedan configurar los parámetros de la red).

## Características del Modelo Matemático
El modelo evalúa un escenario de red compleja teniendo en cuenta:
* **Costes Fijos:** Apertura de almacenes, uso de camiones (Orígenes -> Almacenes) y uso de furgonetas (Almacenes -> Destinos).
* **Costes Variables:** Coste unitario de transporte por arista.
* **Restricciones:** Satisfacción estricta de la demanda, límites de oferta, conservación de flujo en los transbordos (balance de masas), y capacidades máximas tanto de los almacenes como de los vehículos (con variables de decisión enteras y binarias).
