# Módulo: `data_loader.py`

Utilidad simple para crear el generador de test a partir de la configuración.

## Función: `create_test_generator(config) -> DirectoryIterator`

Parámetros leídos desde `config`:

- `data.test_dir`
- `data.image_size`
- `data.batch_size`

Regresa un `ImageDataGenerator(...).flow_from_directory(...)` con `rescale=1/255` y `shuffle=False`.
