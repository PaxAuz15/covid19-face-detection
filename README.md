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
*  Generate CUDA mex for inference speed acceleration   
 
Part1 - Prepare Data
-------------
<img src="/images/image3.png" width="30%" height="30%" title="Ground Truth Labeling" alt="Ground Truth Labeling"></img><br/>

#### COVID19_LabelMaskAutomation.mlx
This file includes the basic of ground truth labeling and how we semi-automate the ground truth labeling with pretrained open source model.

Part2 - Train Model
-------------
#### COVID19_TrainMaskDetection.mlx
This file includes the entire training process from data augmentation, architecture creation and evaluation.
It includes high-level APIs for SSD(Single Shot Multibox Detector) and YOLOv2(You Only Look Once) network architecture for the comparison.

If you complete the training, we need to test the trained model for still image, video and live webcam streaming data.
For the each task instances, see below codes for the model running.
#### COVID19_TestStillImage.mlx
- Test trained model for still image.
#### COVID19_VideoRunning.mlx
- Test trained model for existing video.
#### COVID19_LiveWebcamMask.mlx
- Test trained model for live webcam object image. The example requires MATLAB Support Package for USB Webcams. If you do not have the required support packages installed, then the software provides a download link.

Part3 - Deploy System
-------------
<img src="/images/image4.png" width="70%" height="70%" title="Inference Speed Comparision" alt="Inference Speed Comparision"></img><br/>

### COVID19_TrainMaskDetection.mlx
In the training code, few lines of code is included for code generation.
Prerequisites
- CUDA enabled NVIDIA GPU with compute capability 3.2 or higher.
- NVIDIA CUDA toolkit and driver.
- NVIDIA cuDNN library.
- Environment variables for the compilers and libraries. For information on the supported versions of the compilers and libraries, see [Third-party Products](https://www.mathworks.com/help/releases/R2020a/gpucoder/gs/install-prerequisites.html#mw_aa8b0a39-45ea-4295-b244-52d6e6907bff) (GPU Coder). For setting up the environment variables, see [Setting Up the Prerequisite Products](https://www.mathworks.com/help/releases/R2020a/gpucoder/gs/setting-up-the-toolchain.html) (GPU Coder).
- GPU Coder Interface for Deep Learning Libraries support package. To install this support package, use the Add-On Explorer.

Requires
- [MATLAB](https://www.mathworks.com/products/matlab.html)
- [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html)
- [Image Processing Toolbox](https://www.mathworks.com/products/image.html)
- [Computer Vision Toolbox](https://www.mathworks.com/products/computer-vision.html)
- [Parallel Computing Toolbox](https://www.mathworks.com/products/parallel-computing.html)
- [MATLAB Coder](https://www.mathworks.com/products/matlab-coder.html)
- [GPU Coder](https://www.mathworks.com/products/gpu-coder.html)

Support Packages
- [Deep Learning Toolbox Importer for Caffe Models](https://www.mathworks.com/matlabcentral/fileexchange/61735-deep-learning-toolbox-importer-for-caffe-models)
- [MATLAB Support Package for USB Webcams](https://www.mathworks.com/matlabcentral/fileexchange/45182-matlab-support-package-for-usb-webcams)
- [GPU Coder Interface for Deep Learning Libraries](https://kr.mathworks.com/matlabcentral/fileexchange/68642-gpu-coder-interface-for-deep-learning-libraries)


For more information on Deep Learning in MATLAB
-------------
[![View COVID19-Face-Mask-Detection-using-deep-learning on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://kr.mathworks.com/matlabcentral/fileexchange/76758-covid19-face-mask-detection-using-deep-learning)

**[Download a free MATLAB trial for Deep Learning](https://www.mathworks.com/products/deep-learning.html)**

[View Webinar for the entire model development (Korean)](https://www.youtube.com/watch?v=EwCWgsjzR9E)

Copyright 2020 The MathWorks, Inc.
