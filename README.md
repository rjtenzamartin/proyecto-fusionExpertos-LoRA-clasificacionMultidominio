# Fusión de Expertos LoRA para Clasificación Multidominio 🧠⚖️🏥

Este proyecto evalúa si la fusión matemática de modelos especializados (expertos) mediante la técnica LoRA supera el rendimiento de un modelo generalista en tareas de clasificación de textos de dominios complejos. Se centra en dos ámbitos con vocabularios diametralmente opuestos: el médico y el legal.

## 🚀 Características y Metodología

* **Procesamiento de Datos y Chunking:** Para gestionar los extensos documentos legales frente a la restricción de 512 tokens de BERT, se implementó una estrategia de segmentación por ventana deslizante (Chunking) con solapamiento, evitando la pérdida de contexto.


* **Fase 1: Preentrenamiento Adaptativo al Dominio (DAPT):** Se entrenaron dos adaptadores LoRA independientes sobre un modelo `bert-base-uncased` congelado. Este entrenamiento autosupervisado utilizó Masked Language Modeling (MLM) para que cada experto aprendiera la terminología y los matices de su dominio.


* **Fase 2: Fusión y Ajuste Fino Supervisado (SFT):** Los pesos de ambos adaptadores especializados se fusionaron aritméticamente (combinación lineal con un coeficiente de mezcla de 0.5). Posteriormente, se añadió una cabeza de clasificación densa para entrenar el modelo en una tarea supervisada de 7 clases.



## 📊 Resultados Destacados

* **Rendimiento Predictivo:** El modelo de fusión de expertos superó al modelo generalista en la métrica F1-Score ponderado, demostrando que la especialización modular preserva mejor los matices semánticos sin las interferencias del entrenamiento conjunto.


* **Eficiencia Computacional:** Gracias al uso de adaptadores LoRA (rango 16), el consumo máximo de memoria de vídeo (VRAM) se mantuvo por debajo de los 7 GB (aprox. 6.26 GB en la fase más exigente), permitiendo el entrenamiento de modelos masivos en hardware accesible como una GPU NVIDIA T4.


* **Representación Latente (T-SNE):** Las proyecciones vectoriales confirmaron que la arquitectura de expertos logra una organización del espacio latente superior, mostrando una separación geográfica nítida entre las clases médicas y legales.


* **Escalabilidad Continua:** El sistema permite añadir nuevos dominios en el futuro entrenando simplemente un nuevo adaptador liviano y promediando sus pesos, sin necesidad de reentrenar todo el sistema base.



## 🛠️ Tecnologías Utilizadas

* **Modelo Base:** BERT (`bert-base-uncased`).


* **Técnicas PEFT:** LoRA (Low-Rank Adaptation).


* **Frameworks:** Hugging Face (Transformers, PEFT), PyTorch, Scikit-learn.
