   # Dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   # Copiar requisitos y instalarlos
   COPY requirements.txt .
   RUN pip install --upgrade pip && pip install -r requirements.txt

   # Copiar todo el código fuente
   COPY . .

   # Cambiar el directorio de trabajo a src para que se encuentre la estructura correcta
   WORKDIR /app/src

   # Exponer el puerto 8080 (Cloud Run lo usa por defecto)
   EXPOSE 8080

   # Comando para iniciar la aplicación
   CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]