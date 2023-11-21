# Informe de VC-P5.

Este proyecto se enfoca en la detección y extracción de matrículas vehiculares mediante la exploración de diversos enfoques. Se aborda desde distintas perspectivas, empleando una variedad de técnicas y herramientas para identificar las placas de vehículos y extraer el texto presente en estas. A lo largo del desarrollo de la práctica, se han implementado múltiples enfoques con el objetivo de analizar las diferencias entre ellos, así como las ventajas y desafíos que han surgido en el proceso. A continuación, se detallan los distintos enfoques implementados y se examinan las experiencias obtenidas al utilizar cada uno de ellos.

## Reconocimiento óptico de caracteres.

Antes de comenzar con las distintas implementaciones del problema, cabe a destacar, que he usado la misma lógica (En una implementación hay un pequeño ajuste) en todas ellas para lograr reconocer y mostrar el texto, en este caso, las letras y números de la matricula por pantalla.

Como ya sabemos, la práctica se puede "simplificar" y dividir en 2 bloques, la lógica de reconocimiento de matrícula y la lógica para la detección y reconocimiento de caracteres a través del uso de la librería _easyocr_ (En esta práctica se ha usado esta librería por su simpleza).

De este modo, en este apartado tal como adelanta su título nos centraremos exclusivamente en mi implementación para reconocer y mostrar las matrículas.

**1. Inicialización del lector de OCR (Reconocimiento Óptico de Caracteres)**

reader = easyocr.Reader(['en'], gpu=False)

- **Descripción:** Esta línea inicializa el lector de EasyOCR para el idioma inglés ('en'). Se configura con la GPU desactivada (gpu=False), lo que significa que se utilizará la CPU para el procesamiento.

**2. Diccionarios de mapeo para conversión de caracteres**

dict\_char\_to\_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'B': '3'}

dict\_int\_to\_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

- **Descripción:** Estos diccionarios proporcionan mapeos para convertir ciertos caracteres en otros. Por ejemplo, 'O' se convierte en '0', 'I' en '1', etc. Son utilizados para normalizar y corregir posibles errores de reconocimiento de caracteres. Esto se debe a que easyOcr es bastante sensible a ciertos datos, y da falsas lecturas, como sabemos el formato de matrícula y los datos sensibles. Podemos recogerlos y corregir como comenté estos errores.

**3. Funciones para verificación y formato de la matrícula**

def license\_complies\_format(text):

# ... (verifica si la matrícula cumple con el formato requerido)

return True / False

def format\_license(text):

# ... (formatea el texto de la matrícula utilizando los mapeos definidos)

return formatted\_license\_plate

**Descripción:**

- **license\_complies\_format** verifica si el texto de la matrícula cumple con el formato requerido (7 caracteres, combinación específica de letras y números, siguiendo el formato de matrículas de vehículos en general).
- **format\_license** formatea el texto de la matrícula utilizando los mapeos definidos en los diccionarios, asegurando una consistencia y corrección en los caracteres.

**4. Función para leer el texto de la matrícula**

def read\_license\_plate(license\_plate\_crop):

# ... (utiliza EasyOCR para leer el texto de la matrícula)

return formatted\_license\_plate, confidence\_score

- **Descripción:** read\_license\_plate toma una imagen recortada que contiene una matrícula y utiliza EasyOCR para leer el texto presente en ella. Luego, verifica si el texto leído cumple con el formato requerido y lo formatea utilizando las funciones previamente definidas. Retorna el texto formateado y la puntuación de confianza del reconocimiento.

**Nota:** Hay que destacar 1 cosa, y es que hay que tener muchísimo cuidado en este punto. Y aquí es donde quiero comentar uno de los pequeños cambios que adelante en el encabezado de reconocimiento de texto. Cuando uno le pasa por parámetro al readtext() de ocr una imagen grande, hay que tener en cuenta de que al ser grande, el espacio que hay ente letras y números, antes despreciable, toma mucha importancia, ya que las lecturas las hará a trozos, es decir, te detecta 2 textos, los números, y luego las letras. Esto hay que tenerlo en cuenta a la hora de procesar el texto, ya que, cuando vayamos a verificar el formato, nos dará error y finalmente la función devolverá None en una matrícula correcta. Para evitar esto, se concatena los 2 textos, y se verifica el formato de estos. Como solo le estamos pasando una imagen de la matricula, no hay problema alguno en concatenar los textos, ya que no hay ruido que distorsione los resultados como, por ejemplo, un cartel de fondo que ponga, STOP. Esta lógica solo se usó en la implementación de las imágenes del dataset, para el correcto funcionamiento de este. Ya que, al ser imágenes tomadas a conciencia para entrenar al modelo de YOLO posteriormente, son menos naturales que, por ejemplo, los frames que podemos encontrarnos en el video. Sin embargo, es destacable que esta lógica se puede implementar en las demás aproximaciones, ya que, como comenté siempre le paso una imagen donde hay un recorte donde solo se ve la matrícula, evitando, como ya dije ruido y datos que realmente no me importan. Sin embargo, al ser un trabajo de entorno académico considere oportuno, solo hacerlo en una aproximación para poder destacar este dato, ya que esta pequeña anotación añade robustez y consistencia al código en todas las aproximaciones.

**5. Uso del reconocimiento de matrículas en el código principal**

license\_plate\_text, license\_plate\_text\_score = read\_license\_plate(license\_plate\_crop)

# ... (se usa el texto de la matrícula en el procesamiento principal)

if license\_plate\_text is not None:

cv2.putText(roi, license\_plate\_text, (x, y - 20), cv2.FONT\_HERSHEY\_SIMPLEX, 1, (0, 0, 255), 2)

- **Descripción:** En el flujo principal del código, se lee el texto de la matrícula usando la función read\_license\_plate. Si se detecta una matrícula válida, se coloca el texto de la matrícula en la imagen principal utilizando OpenCV.

**Nota:** Aquí lo "guapo" es jugar con el parámetro que le pasas a la función, personalmente la misma imagen aplicándole diferentes procesamientos para ver, cuales daban las mejores lecturas. Probé muchísimas cosas, aplicarle diferentes umbrales, ya sea con escala de grises o umbralización binaria, etc. Siempre jugando con los parámetros para buscar y facilitar una mejor lectura al easyocr. Para asegurar la diversidad y que usted vea, que efectivamente he probado todo lo que le comento. En el código podrá ver que use diferentes umbrales, y incluso en el video no use ninguno (Como hablamos en una tutoría). Además de meterle umbrales, utilicé también el resize para ampliar y agrandar la imagen recortada de la matricula y amplie el recorte para que fuese más limpio.

Sin embargo, hay que tener mucho cuidado en esto último, ya que, si no tenemos cuidado, nos estamos enfrentando a este error.

_Error al redimensionar la imagen: OpenCV(4.8.1) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\resize.cpp:4062: error: (-215:Assertion failed) !ssize.empty() in function 'cv::resize'_

El cual da quebraderos de cabeza. Pero tiene "simple" solución, y es comprobando que los valores sean correctos antes de hacer nada.

## Reconocimiento de matrículas.

La práctica aborda dos enfoques diferentes para la detección de matrículas: uno manual basado en el análisis de contornos y otro basado en un modelo preentrenado de YOLOv8. Se comparan y contrastan sus resultados en escenarios individuales, conjuntos de imágenes y videos en tiempo real.

**Enfoque de Detección Manual mediante Análisis de Contornos**

Este enfoque se centra en el análisis de contornos para identificar formas rectangulares, típicamente asociadas con matrículas. Se emplean técnicas de procesamiento de imágenes para detectar y extraer regiones con formas aproximadas a rectángulos, seguido de una validación adicional para confirmar si estas regiones corresponden a matrículas.

**Caso de Estudio 1: Evaluación en Imagen Individual**

Se aplicó el algoritmo de detección manual en una imagen estática para analizar su desempeño en la identificación de matrículas. Este caso de estudio realmente sirve para preparar al algoritmo para que se comporte mejor de cara al siguiente caso de estudio.

**Conclusión del caso 1**

Me sirvió para calibrar los parámetros de los umbrales como los de validación para obtener resultados decentes.

**Caso de Estudio 2: Evaluación en un Conjunto de Imágenes (Dataset)**

El algoritmo se sometió a pruebas en un conjunto diverso de imágenes que simulan escenarios del mundo real. Se evaluó su robustez frente a variaciones en iluminación, orientación, y condiciones ambientales diversas.

**Conclusión del caso 2.**

En esta evaluación es donde realmente vemos las flaquezas del modelo de detección, ya que, al mínimo cambio en la imagen, se han de ajustar los parámetros de los umbrales acorde para minimizar el ruido y poder obtener la rectangularidad de manera más limpia, y esto se ve reflejado en que hay algunas imágenes cuya matricula es clara, y a pesar de eso, el algoritmo no se adapta bien a la iluminación y es incapaz de ver la matricula.

Destacando, de este modo, que para cada imagen, habría que manipular los parámetros de manera individual para que poder hallar la matricula lo cual es logísticamente un esfuerzo enorme.

**Ejemplos.**

**Detección negativa.**

<img src="lalaland.jpg" width="300" height="200">

 **Detección positiva.**

<img src="ryan-gosling-1655362981.jpg" width="300" height="200">


**Enfoque de Detección con YOLOv8 (Modelo Preentrenado)**

Se utilizó un modelo preentrenado de YOLOv8, utilizando un dataset sacado de roboflow [https://universe.roboflow.com/licenseplates-h9qfr/spanish-license-plates/dataset/2](https://universe.roboflow.com/licenseplates-h9qfr/spanish-license-plates/dataset/2)

**Caso de Estudio 3: Evaluación en Imagen Individual con YOLOv8**

Se empleó el modelo preentrenado de YOLOv8 en una imagen individual para comparar su rendimiento con el enfoque de detección manual.

**Conclusión del caso 3**

No hay mucho que destacar, ya de por si YOLOv8 te genera una carpeta con los resultados del entrenamiento, este caso de estudio sirvió para verificar el correcto funcionamiento del detector. El cual ha demostrado servir perfectamente.

**Caso de Estudio 4: Evaluación en Video Tiempo Real**

El modelo de YOLOv8 fue aplicado en un video en tiempo real para observar su capacidad de detección dinámica de matrículas en movimiento.

**Conclusión del caso 4.**

Para mi sorpresa mi modelo funcionó perfectamente, y no hubo que ajustar nada en particular. El problema en la evaluación en Video en Tiempo real reside en el reconocimiento óptico de imágenes, esto lo comentaré más adelante en el informe. Volviendo al modelo, este se ha adaptado muy bien a este caso de estudio, es más, es tan eficiente, que no le hace falta tracking para poder seguir la matricula a lo largo de los fotogramas.

**Caso de Estudio 5: Preprocesamiento de Video con YOLOv8**

Se permitió al modelo de YOLOv8 realizar el preprocesamiento del video antes de la detección, lo que incluyó operaciones de mejora de calidad y estabilización de imágenes, para determinar si estos pasos mejoran la precisión en la detección de matrículas.

Cabe a destacar que este último, se veía asombrosamente mal. Personalmente me parecía una chapuza entregarle eso, así que tras informarme vi que usando técnicas de interpolación el output era más fluido.

**Nota:** Todo esto tiene un por qué, sin embargo, ya en diferentes tutorías le he hablado de ello, de manera que volvérselo a comentar es redundante, si usted lo desea, puedo agregarlo.

##

## Bibliografía.

https://www.cea-online.es/blog/774-tipos-de-matriculas-de-vehiculos-en-espana
