# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="DBpAkqcQiMIU"
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sensioai/blog/blob/master/027_pytorch_intro/pytorch_intro.ipynb)

# %% [markdown] id="hPioFv-kiMIY"
# # Pytorch - Introducción

# %% [markdown] id="6cYk2MxQiMIZ"
# Hast ahora hemos implementado nuestros propios modelos de `Machine Learning` utilizando `Python` y `Numpy`. Este ejercicio nos ha servido para aprender mejor a utilizar estas herramientas de `análisis de datos` así como a conocer en gran detalle algunos de los elementos fundamentales de las redes neuronales: el [perceptrón](https://sensioai.com/blog/018_perceptron_final), el algoritmo de [descenso por gradiente](https://sensioai.com/blog/013_perceptron2), el [perceptrón multicapa](https://sensioai.com/blog/025_mlp_framework), etc. Sin embargo, de ahora en adelante, utilizaremos frameworks desarrollados por terceros. En el [post anterior](https://sensioai.com/blog/026_frameworks) hablamos en detalle de los motivos y presentamos algunos ejemplos. En este post, empezamos a aprender a trabajar con uno de los frameworks de `redes neuronales` más utilizados hoy en día: [Pytorch](https://pytorch.org/).

# %% [markdown] id="ObsNSgx0iMIa"
# > ⚡ Si trabajas en Google Colab ya tendrás `Pytorch` instalado. Si quieres trabajar en local, simplemente sigue las instrucciones en https://pytorch.org/. Te recomiendo la instalación con `conda`, sobre todo si quieres soporte para GPU.

# %% id="SqAvfVjliMIa" executionInfo={"status": "ok", "timestamp": 1774454300580, "user_tz": 240, "elapsed": 1, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
import torch

# %% [markdown] id="y6bKEKnhiMIb"
# ## ¿ Qué es Pytorch ?

# %% [markdown] id="UcT_BOWHiMIc"
# `Pytorch` es un framework de `redes neuronales`, un conjunto de librerías y herramientas que nos hacen la vida más fácil a la hora de diseñar, entrenar y poner en producción nuestros modelos de `Deep Learning`. Una forma sencilla de entender qué es `Pytorch` es la siguiente:
#
# $$ Pytorch = Numpy + Autograd + GPU $$
#
# Vamos a ver qué significa cada uno de estos términos.

# %% [markdown] heading_collapsed=true id="v-_MT-I2iMId"
# ## NumPy

# %% [markdown] hidden=true id="UJBccc8fiMIe"
# Quizás la característica más relevante de `Pytorch` es su facilidad de uso. Esto es debido a que sigue una interfaz muy similar a la de `NumPy`, y como nosotros ya sabemos trabajar con esta librería no deberíamos tener muchos problemas para aprender a trabajar con `Pytorch` 😁.

# %% [markdown] hidden=true id="jGFnEVcNiMIe"
# > 🧠 Si no estás familiarizado con la librería `NumPy` te recomiendo que le eches un vistazo a nuestros posts en los que aprendemos a trabajar con esta librería.

# %% [markdown] hidden=true id="vo4X5a2niMIf"
# De la misma manera que en `NumPy` el objeto principal es el `ndarray`, en `Pytorch` el objeto principal es el `tensor`. Podemos definir un tensor de manera similar a como definimos un array, incluso podemos inicializar tensores a partir de arrays.

# %% hidden=true id="o_ERwwGkiMIf" outputId="138aedfe-fe29-4fc5-cd6f-fe862f99daa3" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454300628, "user_tz": 240, "elapsed": 40, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
import numpy as np

# matriz de ceros, 5 filas y 3 columnas
xn = np.zeros((5, 3))
x = torch.zeros(5, 3)

print(xn, x)

# %% hidden=true id="lcbVUMRdiMIg" outputId="80f104d8-05a3-4134-ff71-6e42df45fa2d" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454300633, "user_tz": 240, "elapsed": 5, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# tensor con valores aleatorios

x = torch.randn(5, 3, 2)
print(x)

# %% hidden=true id="ULZfDNL0iMIg" outputId="db1f2fa1-6a78-4004-d906-cf97351ab651" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454300652, "user_tz": 240, "elapsed": 8, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# tensor a partir de lista

x = torch.tensor([[1, 2, 3],[4, 5, 6]])
print(x)
# %% hidden=true id="-hBxpgUIiMIh" outputId="2f4d5b7b-059b-4a27-84e8-a5ef2cb91a21" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454300656, "user_tz": 240, "elapsed": 3, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
import numpy as np

# tensor a partir de array

a = np.array([[1, 2, 3],[4, 5, 6]])
x = torch.from_numpy(a)
print(x)
print(a)

# %% [markdown] hidden=true id="kfK89vQ3iMIh"
# Y como puedes esperar, prácticamente todos los conceptos que ya conocemos para trabajar con `NumPy` pueden aplicarse en `Pytorch`. Esto incluye operaciones aritméticas, indexado y troceado, iteración, vectorización y broadcasting.

# %% hidden=true id="GmxyiVMMiMIh" outputId="167c03a2-1368-4189-e402-936ccca9752d" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454300677, "user_tz": 240, "elapsed": 20, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# operaciones

x = torch.randn(3, 3)
y = torch.randn(3, 3)

print(x)
print(y)

# %% hidden=true id="gn1z1FOviMIi" outputId="9910bac5-05fa-4fc3-adcd-e4e6ea119912" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454300678, "user_tz": 240, "elapsed": 18, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
print(x + y)

# %% hidden=true id="CqMY2660iMIi" outputId="91b4f3b2-bb9b-4a7f-818d-310ef506d1de" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454300687, "user_tz": 240, "elapsed": 10, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
print(x - y)

# %% hidden=true id="pjYd-XwiiMIi" outputId="e018d4d0-7194-47f6-fda6-cab1d0a8acca" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454300689, "user_tz": 240, "elapsed": 2, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# indexado

# primera fila

print(x[0])

# %% hidden=true id="dYGdMkDmiMIj" outputId="88cf47b1-26bd-467f-837f-434841507b41" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454300706, "user_tz": 240, "elapsed": 16, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# primera fila, primera columna
print(a[0][0])
print(x[0, 0])
# %% hidden=true id="Me8oHUGaiMIj" outputId="94b2547c-8d07-46a0-f18d-15fa6ff51b00" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454300709, "user_tz": 240, "elapsed": 3, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# primera columna

print(x[0, :])

# %% hidden=true id="fw3GdsJ-iMIj" outputId="c29755c6-ed11-4771-95c4-2aa2af364637" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454300737, "user_tz": 240, "elapsed": 28, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# troceado

print(x[:-1, 1:])

# %% [markdown] hidden=true id="smDT-oUaiMIk"
# Una funcionalidad importante del objeto `tensor` que utilizaremos muy a menudo es cambiar su forma. Esto lo conseguimos con la función `view`.

# %% hidden=true id="ovWsFTZYiMIk" outputId="317cb1dc-8102-4631-b47d-2dd85423f970" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454300738, "user_tz": 240, "elapsed": 26, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
print(x.shape)

# %% hidden=true id="aVXMJ_pwiMIk" outputId="33517d3a-8c76-4510-f64a-35226f9cb4c9" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454300738, "user_tz": 240, "elapsed": 4, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# añadimos una dimensión extra

x.view(1, 3, 3).shape

# %% colab={"base_uri": "https://localhost:8080/"} id="LDHzJl0rSmyb" executionInfo={"status": "ok", "timestamp": 1774454300755, "user_tz": 240, "elapsed": 3, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="8250642c-c01f-4507-f250-d7b00df498a7"
x

# %% hidden=true id="IBFSHcL0iMIk" executionInfo={"status": "ok", "timestamp": 1774454300757, "user_tz": 240, "elapsed": 2, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# estiramos en una sola dimensión

# a = x.view(9).shape
a = x.view(9)

# %% colab={"base_uri": "https://localhost:8080/"} id="GcEVwOqMSxg7" executionInfo={"status": "ok", "timestamp": 1774454300763, "user_tz": 240, "elapsed": 3, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="3dd31925-e721-4888-eaf5-0897f7e41d2f"
a

# %% hidden=true id="YoO2CeSjiMIl" executionInfo={"status": "ok", "timestamp": 1774454510226, "user_tz": 240, "elapsed": 3, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# usamos -1 para asignar todos los valores restantes a una dimensión

b = x.view(2, 2, -1).shape

# %% colab={"base_uri": "https://localhost:8080/"} id="ym4NzOf7S2vx" executionInfo={"status": "ok", "timestamp": 1774454512868, "user_tz": 240, "elapsed": 6, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="11dd0b77-13ff-471a-8c95-3ce13408691f"
b

# %% [markdown] hidden=true id="90DuaFxNiMIl"
# Podemos transformar un `tensor` en un `array` con la función `numpy`.

# %% hidden=true id="dKtMQ6t-iMIl" outputId="1021371f-ac4c-4386-d90b-ac60c3a4469e" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454300794, "user_tz": 240, "elapsed": 8, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
x.numpy()

# %% [markdown] hidden=true id="q78bNXIYiMIl"
# Como puedes ver, un `tensor` de `Pytorch` es muy similar a un `array` de `NumPy`. Aquí hemos visto alguna de la funcionalidad más útil, puedes aprender más [aquí](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py).

# %% [markdown] heading_collapsed=true id="cxKGPXIriMIl"
# ## Autograd

# %% [markdown] hidden=true id="WCEX51pfiMIm"
# Ya hemos visto que `Pytorch` es muy similar a `NumPy`, sin embargo su funcionalidad va más allá de una estructura de datos eficiente con la que podemos llevar a cabo operaciones (para eso ya nos basta con `NumPy`). La funcionalidad más importante que `Pytorch` añade es la conocidad como `autograd`, la cual nos proporciona la posibilidad de calcular derivadas de manera automática con respecto a cualquier `tensor`. Esto le da a `Pytorch` un gran potencial para diseñar `redes neuronales` complejas y entrenarlas utilizando algoritmos de gradientes sin tener que calcular todas estas derivadas manualmente (como hemos hecho en los posts anteriores). Para poder llevar a cabo estas operaciones, `Pytorch` va construyendo de manera dinámica un `grafo computacional`. Cada vez que aplicamos una operación sobre uno o varios tensores, éstos se añaden al `grafo computacional` junto a la operación en concreto. De esta manera, si queremos calcular la derivada de cualquier valor con respecto a cualquier tensor, simplemente tenemos que aplicar el algoritmo de `backpropagation` (que no es más que la regla de la cadena de la derivada) en el `grafo`. Vamos a ilustrarlo con un ejemplo.

# %% hidden=true id="63iykJNuiMIm" executionInfo={"status": "ok", "timestamp": 1774454300808, "user_tz": 240, "elapsed": 13, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
x = torch.tensor(1., requires_grad=True)
y = torch.tensor(2., requires_grad=True)
p = x + y

z = torch.tensor(3., requires_grad=True)
g = p * z

# %% [markdown] hidden=true id="pD-t_oWmiMIm"
# En la celda anterior hemos definido tres `tensores`: $x$, $y$ y $z$. En primer lugar, para poder calcular derivadas con respecto a estos tensores necesitamos ponder su propiedad `requiers_grad` a `True`. Ahora, calculamos el tensor intermedio $p$ como $p = x+ y$ y luego usamos este valor para calcular el resultado final $g$ como $g = p*z$. Cada vez que aplicamos una operación sobre un tensor que tiene su propiedad `requires_grad` a `True`, `Pytorch` irá construyendo el `grafo computacional`. Para este ejemplo, el grafo tendría la siguiente forma
#
# ![](https://www.tutorialspoint.com/python_deep_learning/images/computational_graph_equation2.jpg)
#
# Si ahora queremos calcular las derivadas de $g$ con respecto a $x$, $y$ y $z$, es tan fácil como llamar a la función `backward`.

# %% hidden=true id="YtSCg3BZiMIm" executionInfo={"status": "ok", "timestamp": 1774454300817, "user_tz": 240, "elapsed": 8, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
g.backward()

# %% [markdown] hidden=true id="CcUUp2HmiMIn"
# En este punto, `Pytorch` ha aplicado el algoritmo de `backpropagation` encima del grafo computacional, calculando todas las derivadas.
#
# $$ \frac{dg}{dz} = p $$

# %% hidden=true id="OTYZAqWViMIn" outputId="641518ea-f253-44ae-fb2e-de4bfc977a7b" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454300818, "user_tz": 240, "elapsed": 3, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
z.grad

# %% [markdown] hidden=true id="EMoUWf8ZiMIn"
# $$ \frac{dg}{dx} = \frac{dg}{dp} \frac{dp}{dx} = z $$

# %% hidden=true id="BSY3Sm7ViMIn" outputId="f75145aa-75ae-420d-c9c7-acf2316c92f2" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454300838, "user_tz": 240, "elapsed": 21, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
x.grad

# %% [markdown] hidden=true id="GDlvvVJ8iMIn"
# $$ \frac{dg}{dy} = \frac{dg}{dp} \frac{dp}{dy} = z $$

# %% hidden=true id="ByIg6BDyiMIo" outputId="ccb08fcf-813c-42aa-dad5-03a22acae81a" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454300838, "user_tz": 240, "elapsed": 16, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
y.grad

# %% [markdown] hidden=true id="xyYPdaz4iMIo"
# Como puedes ver, el `grafo computacional` es una herramienta extraordinaria para diseñar `redes neuronales` de complejidad arbitraria. Con una simple función, gracias al algoritmo de `backpropagation`, podemos calcular todas las derivadas de manera sencilla (cada nodo que representa una operación solo necesita calcular su propia derivada de manera local) y optimizar el modelo con nuestro algoritmo de gradiente preferido.

# %% [markdown] hidden=true id="mDO1mmwviMIo"
# > 💡 Sabiendo que el `perceptrón` leva a cabo la operación $\hat{y} = f(\mathbf{w} \cdot \mathbf{x} + b)$, ¿te ves capaz de dibujar su grafo computacional?

# %% [markdown] hidden=true id="6VsOU5uAiMIo"
# Añadiendo `autograd` encima de `NumPy`, `Pytorch` nos ofrece todo lo que necesitamos para diseñar y entrenar `redes neuronales`. Puedes aprender más sobre `autograd` [aquí](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py). Sin embargo, si queremos entrenar redes muy grandes o utilizar datasets muy grandes (o ambas), el proceso de entrenamiento será muy lento. Es aquí donde entra en juego el último elemento que hace de `Pytorch` lo que es.

# %% [markdown] heading_collapsed=true id="sLsBkDY6iMIo"
# ## GPU

# %% [markdown] hidden=true id="YwBCWTJGiMIo"
# Si has prestado atención durante nuestro viaje a través de las diferentes implementaciones que hemos llevado a cabo en los posts anteriores, te habrás dado cuenta que, en su mayoría, nuestros modelos llevan a cabo una operación simple: el producto de matrices. Esta operación puede ser muy lenta si estas matrices son muy grandes. Sin embargo, existe hardware especializado en acelerar precisamente este tipo de operaciones: las unidades de procesado gráfico, o GPUs.
#
# ![](https://media.metrolatam.com/2019/08/05/cqierczlxyy4lox3-759570a83e907b6bfd77a3964c65fec3.jpg)
#
# Si eres *gamer* hay poco sobre este tipo de hardware que no sepas. Para el resto, este chip (que puedes entender como un mini-ordenador dentro de tu ordenador) fue diseñado con el objetivo de acelerar los cálculos necesarios para renderizar una escena tridimensional en la pantalla de tu ordenador. Estas escenas se representan mediante triángulos con una posición determinada en el mundo virtual que se desea representar, y en cada fotograma se tiene que calcular su posición relativa a una cámara virtual, el punto de vista de la cual es renderizado en tu pantalla. Si estas escenas tienen muchos triángulos, hacer estos cálculos en la CPU (la unidad de procesado central de tu ordenador) pueden llevar mucho tiempo, destruyendo la experiencia en tiempo real que los videojuegos requieren. Es por este motivo que utilizamos GPUs, hardware especializado en llevar a cabo estas operaciones de manera rápida permitiendo las experiencias fluidas a las que estamos acostrumbrados hoy en día. Da la casualidad que el tipo de operaciones necesarias para calcular la posición de estos triángulos es la misma que necesitamos para entrenar nuestras `redes neuronales`: el producto de (grandes) matrices. El uso de GPUs para acelerar el entrenamiento de modelos de `Deep Learning` ha supuesto una gran revolución en la última década, y es uno de los motivos principales de la explosión que estamos viviendo en el aumento de aplicaciones.
#
# ![](https://www.researchgate.net/profile/Tolga_Soyata/publication/322525660/figure/fig2/AS:613882104139786@1523372307595/Inside-a-computer-containing-an-i7-5930K-CPU-10-CPU5-in-Table-31-and-64GB-of-DDR4.png)

# %% [markdown] hidden=true id="05vFZdToiMIo"
# Si trabajas en Google Colab, puedes utilizar una GPU de manera gratuita simplemente cambiando el tipo de `runtime`. Si quieres utilizar una GPU de manera local, tendrás que comprar una e instalarla en tu PC.
#
# `Pytorch` nos permite acelerar las operaciones entre `tensores` de manera muy sencilla. Simplemente tenemos que asegurarnos que nuestros `tensores` viven en una GPU, `Pytorch` se encargará del resto.

# %% hidden=true id="H1ptakV7iMIp" outputId="4d5823e0-3705-4c21-9460-d5748d48ce2c" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454300838, "user_tz": 240, "elapsed": 13, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# comprobar que podemos usar GPU

torch.cuda.is_available()

# %% hidden=true id="-ZsMuw1_iMIp" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454300892, "user_tz": 240, "elapsed": 55, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="4b04a949-7286-4628-9b09-8adb8fce934d"
x = torch.randn(10000,1000)
y = torch.randn(10000,1000)

%time z = x*y
print(z)
# %% hidden=true id="jKPXsZhOiMIp" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454301046, "user_tz": 240, "elapsed": 151, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="d87d2db8-af0f-485d-d45d-9c788065d6b9"
x = torch.randn(10000,1000).cuda()
y = torch.randn(10000,1000).cuda()

%time z = x*y

# %% [markdown] hidden=true id="yD9nOdqYiMIp"
# Como puedes observar, llevar a cabo operaciones con grandes tensores en una GPU en vez de la CPU puede resultar en una considerable reducción del tiempo de cálculo. Todas las siguientes maneras son válidas para copiar un tensor en una GPU

# %% hidden=true id="zLn-rSj3iMIp" executionInfo={"status": "ok", "timestamp": 1774454301048, "user_tz": 240, "elapsed": 1, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
device = torch.device("cuda")

x = torch.randn((1000,1000), device="cuda")
x = x.cuda()
x = x.to("cuda")
x = x.to(device)

# %% [markdown] hidden=true id="kT6WerKAiMIp"
# Y para volver a copiar un `tensor` de vuelta en la CPU

# %% hidden=true id="mh_R_4HmiMIq" executionInfo={"status": "ok", "timestamp": 1774454301059, "user_tz": 240, "elapsed": 5, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
device = torch.device("cpu")

x = x.cpu()
x = x.to("cpu")
x = x.to(device)

# %% [markdown] heading_collapsed=true id="5FGntiDXiMIq"
# ## Reimplementando nuestro MLP

# %% [markdown] hidden=true id="0oTGG1xDiMIq"
# Para terminar, vamos a poner todos los conceptos que hemos visto juntos en la reimplementación de nuestro modelo de `MLP` con una sola capa oculta que ya conocemos de posts anteriores. Para ello, llevaremos a cabo la tarea de clasificación de imágenes con el dataset MNIST.

# %% hidden=true id="ROqo-XYsiMIq" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454306727, "user_tz": 240, "elapsed": 5630, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="4bb8e644-2578-4a68-cf57-9c04fbb98cc6"
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
X, Y = mnist["data"], mnist["target"]

X.shape, Y.shape

# %% id="igRXPUCVaK2X" executionInfo={"status": "ok", "timestamp": 1774454306728, "user_tz": 240, "elapsed": 0, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
X = X.to_numpy()
Y = Y.to_numpy()

# %% id="VTTpb00jbHXa" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454306741, "user_tz": 240, "elapsed": 10, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="5d6f8fe3-6aee-4714-89df-a715bbdf4f1e"
print(X.shape)

# %% id="WQxYz8E-cyFW" executionInfo={"status": "ok", "timestamp": 1774454306742, "user_tz": 240, "elapsed": 1, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# %matplotlib inline

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

# %% hidden=true id="KLQO5Kt1iMIq" colab={"base_uri": "https://localhost:8080/", "height": 523} executionInfo={"status": "ok", "timestamp": 1774454307276, "user_tz": 240, "elapsed": 527, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="51f4e83d-a3e4-4557-89be-e8bc72fa1505"
r, c = 3, 5
fig = plt.figure(figsize=(2*c, 2*r))
for _r in range(r):
    for _c in range(c):
        plt.subplot(r, c, _r*c + _c + 1)
        ix = random.randint(0, len(X)-1)
        img = X[ix]
        plt.imshow(img.reshape(28,28), cmap='gray')
        plt.axis("off")
        plt.title(Y[ix])
plt.tight_layout()
plt.show()

# %% hidden=true id="KWuWJNmdiMIq" executionInfo={"status": "ok", "timestamp": 1774454307435, "user_tz": 240, "elapsed": 158, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# normalizamos los datos

X_train, X_test, y_train, y_test = X[:60000] / 255., X[60000:] / 255., Y[:60000].astype(np.int64), Y[60000:].astype(np.int64)


# %% hidden=true id="XikttOaCiMIr" executionInfo={"status": "ok", "timestamp": 1774454307437, "user_tz": 240, "elapsed": 1, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# función de pérdida y derivada

def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(axis=-1,keepdims=True)

def cross_entropy(output, target):
    logits = output[torch.arange(len(output)), target]
    loss = - logits + torch.log(torch.sum(torch.exp(output), axis=-1))
    loss = loss.mean()
    return loss


# %% code_folding=[21, 35] hidden=true id="P1gyCb2_iMIr" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454308563, "user_tz": 240, "elapsed": 1125, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="fc3bfd69-c6c1-4917-e5d0-546900b7768d"
D_in, H, D_out = 784, 100, 10

# pesos del MLP (copiamos en gpu)
w1 = torch.tensor(np.random.normal(loc=0.0,
          scale = np.sqrt(2/(D_in+H)),
          size = (D_in, H)), requires_grad=True, device="cuda", dtype=torch.float)
b1 = torch.zeros(H, requires_grad=True, device="cuda", dtype=torch.float)

w2 = torch.tensor(np.random.normal(loc=0.0,
          scale = np.sqrt(2/(D_out+H)),
          size = (H, D_out)), requires_grad=True, device="cuda", dtype=torch.float)
b2 = torch.zeros(D_out, requires_grad=True, device="cuda", dtype=torch.float)

# convertimos datos a tensores y copiamos en gpu
X_t = torch.from_numpy(X_train).float().cuda()
Y_t = torch.from_numpy(y_train).long().cuda()

epochs = 100
lr = 0.8
log_each = 10
l = []
for e in range(1, epochs+1):

    # forward
    h = X_t.mm(w1) + b1
    h_relu = h.clamp(min=0) # relu
    y_pred = h_relu.mm(w2) + b2

    # loss
    # print(y_pred, Y_t)
    # print(y_pred.shape, Y_t.shape)
    loss = cross_entropy(y_pred, Y_t)
    l.append(loss.item())

    # Backprop (calculamos todos los gradientes automáticamente)
    loss.backward()

    with torch.no_grad():
        # update pesos
        w1 -= lr * w1.grad
        b1 -= lr * b1.grad
        w2 -= lr * w2.grad
        b2 -= lr * b2.grad

        # ponemos a cero los gradientes para la siguiente iteración
        # (sino acumularíamos gradientes)
        w1.grad.zero_()
        w2.grad.zero_()
        b1.grad.zero_()
        b2.grad.zero_()

    if not e % log_each:
        print(f"Epoch {e}/{epochs} Loss {np.mean(l):.5f}")


# %% hidden=true id="XbNp1vWSiMIr" executionInfo={"status": "ok", "timestamp": 1774454308565, "user_tz": 240, "elapsed": 1, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
def evaluate(x):
    h = x.mm(w1) + b1
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2) + b2
    y_probas = softmax(y_pred)
    return torch.argmax(y_probas, axis=1)


# %% hidden=true id="T_5ix9cOiMIr" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1774454308592, "user_tz": 240, "elapsed": 21, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="6a5cfab8-575f-4b9f-b0b0-de4096f519c1"
from sklearn.metrics import accuracy_score

y_pred = evaluate(torch.from_numpy(X_test).float().cuda())
accuracy_score(y_test, y_pred.cpu().numpy())

# %% hidden=true id="j7z67UleiMIr" colab={"base_uri": "https://localhost:8080/", "height": 523} executionInfo={"status": "ok", "timestamp": 1774454309117, "user_tz": 240, "elapsed": 518, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="cef35516-296a-48dc-d19d-b2738e1e4aae"
r, c = 3, 5
fig = plt.figure(figsize=(2*c, 2*r))
test_imgs, test_labs = [], []
for _r in range(r):
    for _c in range(c):
        plt.subplot(r, c, _r*c + _c + 1)
        ix = random.randint(0, len(X_test)-1)
        img = X_test[ix]
        y_pred = evaluate(torch.tensor([img]).float().cuda())[0]
        plt.imshow(img.reshape(28,28), cmap='gray')
        plt.axis("off")
        plt.title(f"{y_test[ix]}/{y_pred}", color="green" if y_test[ix] == y_pred else "red")
plt.tight_layout()
plt.show()

# %% [markdown] hidden=true id="eo4b3_0XiMIr"
# Como puedes observar, simplemente definiendo los `tensores` para los pesos y los datos y copiándolos a la GPU podemos definir el `grafo computacional` de manera dinámica aplicando operaciones sobre los tensores (multiplicamos por los pesos y sumamos el *bias*). Una vez tenemos la salida del `MLP` calculamos la función de pérdida y llamando a la función `backward` `Pytorch` se encarga de calcular todas las derivadas de manera automática. Una vez tenemos los gradientes con respecto a los pesos, podemos actualizarlos.

# %% [markdown] id="vmhl06ifiMIs"
# ## Resumen

# %% [markdown] id="4Muo9ZBEiMIs"
# En este post hemos visto una introducción a `Pytorch`, un framework de `redes neuronales` muy utilizado a día de hoy. Hemos visto que `Pytorch` es muy similar a `NumPy` y comparten gran parte de su sintaxis, lo cual es una ventaja si ya sabemos trabajar con `NumPy`. Además, añade `autograd`, la capacidad de construir de manera dinámica un `grafo computacional` de manera que en cualquier momento podemos calcular derivadas con respecto a cualquier tensor de manera automática. Por último, hemos visto como podemos ejecutar todas estas operaciones en una GPU para acelerar el proceso de entrenamiento de nuestros modelos de `Deep Learning`. Este es el núcleo de `Pytorch`, sin embargo esta librería nos ofrece más funcionalidad, de la cual hablaremos más adelante, que nos será muy útil para diseñar, entrenar y poner a trabajar nuestras `redes neuronales`.
