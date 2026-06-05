# Conectar VS Code a Google Colab Pro para usar la GPU

Esta guía detalla los pasos para usar los recursos de cómputo (GPU/TPU/RAM) de tu plan de Google Colab Pro directamente desde la aplicación de escritorio de Visual Studio Code.

## Requisitos Previos (En tu PC)
1. Instalar **Visual Studio Code**.
2. Instalar la extensión **Remote - SSH** de Microsoft en VS Code.

## Paso 1: Configuración en Google Colab Pro

1. Abre un nuevo **Cuaderno (Notebook)** en [Google Colab](https://colab.research.google.com/).
2. Activa la GPU: Ve a **Entorno de ejecución** > **Cambiar tipo de entorno de ejecución** y selecciona un acelerador de hardware (T4, V100, A100, etc.).
3. Monta tu Google Drive (Recomendado para no perder tu código cuando finalice la sesión de Colab):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. En una nueva celda, instala la librería `colab-ssh` y lanza el túnel de Cloudflare:
   ```python
   !pip install colab_ssh --upgrade
   
   from colab_ssh import launch_ssh_cloudflare
   # Reemplaza "tu_contraseña_segura" con una contraseña de tu elección
   launch_ssh_cloudflare(password="tu_contraseña_segura")
   ```
5. Ejecuta la celda. Verás una salida que contiene la información de conexión, prestándole especial atención al comando parecido a: `ssh root@<direccion-aleatoria>.trycloudflare.com`

## Paso 2: Conectar VS Code (Local)

1. En tu VS Code de escritorio, abre la **Paleta de Comandos** (`Ctrl+Shift+P` en Windows/Linux o `Cmd+Shift+P` en Mac).
2. Escribe y selecciona **Remote-SSH: Connect to Host...**
3. Selecciona **+ Add New SSH Host...**
4. Pega el comando `ssh root@...` que te generó Google Colab en el Paso 1 y presiona Enter.
5. Selecciona tu archivo de configuración SSH predeterminado (por ejemplo, `~/.ssh/config` o `C:\Users\tu_usuario\.ssh\config`).
6. Se abrirá un cuadro de diálogo en la parte inferior derecha, haz clic en **Connect** (o búscalo de nuevo en la Paleta de comandos seleccionando el host agregado).
7. VS Code te preguntará algunas cosas la primera vez:
   * **Plataforma:** Selecciona **Linux**.
   * **Fingerprint:** Elige **Continue** (Continuar).
   * **Contraseña:** Ingresa la contraseña ("tu_contraseña_segura") que configuraste en Colab.

## Paso 3: Trabajar en Remoto

1. Una vez conectado, el borde inferior izquierdo de tu VS Code mostrará el nombre del servidor remoto.
2. Ve a **Archivo > Abrir Carpeta...** (`File > Open Folder...`).
3. Ingresa la ruta `/content/` (o `/content/drive/MyDrive/` si quieres ir directo a tus archivos persistentes de Drive).
4. **¡Listo!** Ahora puedes clonar repositorios, abrir terminales y ejecutar scripts de Python aprovechando todo el hardware de Google Colab Pro desde tu editor local.

## Notas y Consideraciones Importantes
* **Sesiones efímeras:** Las sesiones de Colab tienen un tiempo máximo. Si el entorno de Colab se desconecta, tu conexión de VS Code se caerá. Tendrás que regresar al navegador web, reiniciar el entorno y correr el paso 1 nuevamente.
* **Archivos guardados:** Guarda siempre tu código y datasets dentro de `/content/drive/MyDrive/` para evitar que se eliminen cuando la sesión termine. Archivos que guardes solo en `/content/` o directorios raíz se perderán tras desconectarse.
