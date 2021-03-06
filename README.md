Demostración de detección de mascarillas usando DataSets
=============
<img src="/images/image1.png" width="20%" height="20%" title="Resultado Final" alt="Resultado Final"></img><br/>

**Nota**
La licencia para este paquete está disponible en el archivo license.txt. Al ejecutar la secuencia de comandos COVID19_LabelMaskAutomation.mlx, descargará cierto contenido de terceros con licencia bajo acuerdos de licencia separados.

**Antecedentes**

La enfermedad por coronavirus (COVID-19) es una nueva cepa de enfermedad en humanos descubierta en 2019 que nunca se había identificado en el pasado.
El coronavirus es una gran familia de virus que causa enfermedades en pacientes que van desde el resfriado común hasta síndromes respiratorios avanzados como el síndrome respiratorio de Oriente Medio (MERS-COV) y el síndrome respiratorio agudo severo (SARS-COV).
Actualmente, muchas personas se ven afectadas y reciben tratamiento en todo el mundo, lo que provoca una pandemia mundial.
Varios países han declarado una emergencia nacional y han puesto en cuarentena a millones de personas.

Para ser parte de la tendencia mundial, he creado un modelo de aprendizaje profundo de detección de máscaras CORONA.
Incluye etiquetado de datos semiautomático, entrenamiento de modelos y generación de código GPU para inferencia en tiempo real.
Nuestro personal de MathWorks Korea estaba dispuesto a compartir sus selfies (no distribuibles) con máscaras mientras trabajaba desde casa, para que pueda crear el conjunto de datos fácilmente.
Desafortunadamente, el conjunto de datos no se puede distribuir, por lo que debe crear su propio conjunto de datos para entrenar su propio modelo. He incluido algunos datos de muestra en la carpeta SampleMaskData.

Flujo de Trabajo de la Demo
-------------   
<img src="/images/image2.png" width="70%" height="70%" title="Demo Workflow" alt="Demo Workflow"></img><br/>
* Etiquetado de Imagenes  
  * Etiquetado automatizado con modelo preentrenado
  * Utilice Image Labeler para la automatización interactiva de procesos
* Image Labeler para la automatización interactiva de procesos    
  * SSD(Single-Shot Multibox Detector)
  * YOLOv2(You Only Look Once v2)
*  Genere CUDA mex para la aceleración de la velocidad de inferencia  
 
Parte 1 - Preparar la data
-------------
<img src="/images/image3.png" width="30%" height="30%" title="Ground Truth Labeling" alt="Ground Truth Labeling"></img><br/>

#### COVID19_LabelMaskAutomation.mlx
Este archivo incluye lo básico del etiquetado de la verdad del terreno y cómo semi-automatizamos el etiquetado de la verdad del terreno con un modelo de código abierto previamente entrenado.

Part2 - Entrenar el Modelo
-------------
#### COVID19_TrainMaskDetection.mlx
Este archivo incluye todo el proceso de formación desde el aumento de datos, la creación y la evaluación de la arquitectura.
Incluye API de alto nivel para la arquitectura de red SSD (Single Shot Multibox Detector) y YOLOv2 (You Only Look Once) para la comparación.

Si completa la capacitación, debemos probar el modelo entrenado para datos de transmisión de imágenes fijas, videos y cámaras web en vivo.
Para las instancias de cada tarea, consulte los códigos a continuación para el modelo en ejecución.
#### COVID19_TestStillImage.mlx
- Pruebe el modelo entrenado para obtener una imagen fija.
#### COVID19_VideoRunning.mlx
- Pruebe el modelo entrenado para el video existente.
#### COVID19_LiveWebcamMask.mlx
- Pruebe el modelo entrenado para la imagen del objeto de la cámara web en vivo. El ejemplo requiere MATLAB Support Package para cámaras web USB. Si no tiene instalados los paquetes de soporte necesarios, el software proporciona un enlace de descarga.
Part3 - Implementar sistema
-------------
<img src="/images/image4.png" width="70%" height="70%" title="Inference Speed Comparision" alt="Inference Speed Comparision"></img><br/>

### COVID19_TrainMaskDetection.mlx
En el código de entrenamiento, se incluyen algunas líneas de código para la generación de código.
Prerrequisitos
- GPU NVIDIA compatible con CUDA con capacidad informática 3.2 o superior.
- Kit de herramientas y controlador NVIDIA CUDA.
- Biblioteca NVIDIA cuDNN.
- Variables de entorno para los compiladores y bibliotecas. Para obtener información sobre las versiones compatibles de los compiladores y bibliotecas, consulte [Productos de terceros] (https://www.mathworks.com/help/releases/R2020a/gpucoder/gs/install-prerequisites.html#mw_aa8b0a39-45ea- 4295-b244-52d6e6907bff) (Codificador GPU). Para configurar las variables de entorno, consulte [Configuración de los productos de requisitos previos] (https://www.mathworks.com/help/releases/R2020a/gpucoder/gs/setting-up-the-toolchain.html) (Codificador de GPU) .
- Paquete de soporte de interfaz de codificador de GPU para bibliotecas de aprendizaje profundo. Para instalar este paquete de soporte, use el Explorador de complementos.

Requiere
- [MATLAB] (https://www.mathworks.com/products/matlab.html)
- [Caja de herramientas de aprendizaje profundo] (https://www.mathworks.com/products/deep-learning.html)
- [Caja de herramientas de procesamiento de imágenes] (https://www.mathworks.com/products/image.html)
- [Caja de herramientas de visión por computadora] (https://www.mathworks.com/products/computer-vision.html)
- [Caja de herramientas de computación paralela] (https://www.mathworks.com/products/parallel-computing.html)
- [Codificador MATLAB] (https://www.mathworks.com/products/matlab-coder.html)
- [Codificador de GPU] (https://www.mathworks.com/products/gpu-coder.html)

Paquetes de soporte
- [Importador de cajas de herramientas de aprendizaje profundo para modelos Caffe] (https://www.mathworks.com/matlabcentral/fileexchange/61735-deep-learning-toolbox-importer-for-caffe-models)
- [Paquete de soporte de MATLAB para cámaras web USB] (https://www.mathworks.com/matlabcentral/fileexchange/45182-matlab-support-package-for-usb-webcams)
- [Interfaz de codificador de GPU para bibliotecas de aprendizaje profundo] (https://kr.mathworks.com/matlabcentral/fileexchange/68642-gpu-coder-interface-for-deep-learning-libraries)


Para obtener más información sobre Deep Learning en MATLAB
-------------
[! [Ver COVID19-Face-Mask-Detection-using-deep-learning en File Exchange] (https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)] (https: // kr.mathworks.com/matlabcentral/fileexchange/76758-covid19-face-mask-detection-using-deep-learning)

** [Descargue una prueba gratuita de MATLAB para Deep Learning] (https://www.mathworks.com/products/deep-learning.html) **

[Ver seminario web para el desarrollo completo del modelo (coreano)] (https://www.youtube.com/watch?v=EwCWgsjzR9E)

Copyright 2020 The MathWorks, Inc.
