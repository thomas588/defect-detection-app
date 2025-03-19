import os
import cv2
from ultralytics import YOLO
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

class Inference:
    def __init__(self):
        import streamlit as st
        self.st = st

        # Папка для пользовательских моделей
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

        # "Стандартные" модели Ultralytics (например, YOLOv8n)
        self.available_models = [
            x.replace("yolo", "YOLO")
            for x in GITHUB_ASSETS_STEMS
            if x.startswith("yolo")
        ]

        # Пользовательские модели из папки models
        self.user_models = self._list_user_models()

        # Итоговый список выбора моделей
        self.model_options = self.available_models + self.user_models

        # Параметры приложения
        self.model = None
        self.class_names = []
        self.selected_classes = []
        self.selected_indices = []

        # Параметры детекции
        self.conf = 0.50         # Порог достоверности
        self.iou = 0.50          # Порог пересечения (IoU)
        self.enable_trk = False  # Флаг включения трекинга

        # UI элементы для отображения кадров
        self.org_frame = None
        self.ann_frame = None

        # Имя загруженной модели, чтобы не загружать повторно
        self.loaded_model_name = None

        # Параметр камеры (будет выбран в sidebar)
        self.camera_source = None

    def _list_user_models(self):
        """
        Собирает список всех .pt-файлов из папки self.model_dir
        и возвращает их с добавлением пометки " (user)".
        """
        models_list = []
        for filename in os.listdir(self.model_dir):
            if filename.endswith(".pt"):
                base_name = filename.replace(".pt", "")
                display_name = f"{base_name} (user)"
                models_list.append(display_name)
        return models_list

    def web_ui(self):
        """
        Настройка основной страницы приложения.
        """
        self.st.set_page_config(page_title="Defect detection App", layout="wide")
        title_html = """
        <div>
            <h2 style="color:#FF64DA; text-align:center; margin-bottom:10px;">Defect Detection App with AI</h2>
        </div>
        """
            # <iframe src="http://192.168.31.195:4747/video" width="640" height="480" frameborder="0" allowfullscreen></iframe>
        self.st.markdown(title_html, unsafe_allow_html=True)

    def sidebar(self):
        """
        Боковая панель с настройками:
         1) Загрузка пользовательской модели
         2) Выбор модели
         3) Параметры детекции (Confidence, IoU, Tracking)
         4) Выбор типа камеры: локальная или IP-камера
            - Для локальной камеры – выбор индекса (например, 0–4)
            - Для IP-камеры – ввод URL/IP адреса
         5) Выбор классов
         6) Кнопка "Старт" для запуска инференса
        """
        self.st.sidebar.title("Настройки")

        # (1) Загрузка пользовательской модели
        model_file = self.st.sidebar.file_uploader("Загрузите модель (.pt)", type=["pt"])
        if model_file is not None:
            saved_path = os.path.join(self.model_dir, model_file.name)
            with open(saved_path, "wb") as f:
                f.write(model_file.read())
            self.st.sidebar.success(f"Модель '{model_file.name}' загружена!")
            # Обновляем список пользовательских моделей
            self.user_models = self._list_user_models()
            self.model_options = self.available_models + self.user_models

        # (2) Выбор модели
        chosen_model = self.st.sidebar.selectbox("Модель YOLO", self.model_options)

        # (3) Параметры детекции: Confidence, IoU, Tracking
        self.conf = self.st.sidebar.slider("Порог достоверности", 0.0, 1.0, self.conf, 0.01)
        self.iou = self.st.sidebar.slider("Порог пересечения", 0.0, 1.0, self.iou, 0.01)
        self.enable_trk = self.st.sidebar.radio("Включение трекинга", ("Yes", "No"))

        # (4) Выбор типа камеры
        camera_source_type = self.st.sidebar.selectbox("Тип камеры", ("Локальная камера", "IP-камера"))
        if camera_source_type == "Локальная камера":
            # Предлагаем выбрать камеру из набора индексов (например, 0–4)
            camera_index = self.st.sidebar.selectbox("Выберите камеру", [0, 1, 2, 3, 4])
            self.camera_source = camera_index
        else:
            # Для IP-камеры вводится URL или IP адрес
            ip_address = self.st.sidebar.text_input("Введите URL/IP адрес камеры", value="http://")
            self.camera_source = ip_address
            # self.st.write("IP адрес камеры: ", ip_address)
            # self.st.write("IP адрес камеры: ", self.camera_source)

        # (5) Загрузка модели, если выбор изменился
        if chosen_model != self.loaded_model_name:
            self._load_model(chosen_model)
            self.loaded_model_name = chosen_model

        # (6) Если модель загружена, выбор классов для детекции
        if self.model is not None:
            default_classes = self.class_names[:3] if len(self.class_names) >= 3 else self.class_names
            self.selected_classes = self.st.sidebar.multiselect("Выберите классы", self.class_names, default=default_classes)
            self.selected_indices = [self.class_names.index(c) for c in self.selected_classes]

        # Создаём два столбца для отображения кадров
        col1, col2 = self.st.columns(2)
        self.org_frame = col1.empty()
        self.ann_frame = col2.empty()

        # (7) Кнопка "Старт" для запуска инференса
        start_btn = self.st.sidebar.button("Старт")
        return start_btn

    def _load_model(self, chosen_model):
        """
        Загрузка модели:
         - Если выбрана пользовательская модель, берём файл из папки models.
         - Иначе используем стандартную модель (например, YOLOv8n.pt).
        После загрузки обновляем список имен классов.
        """
        if chosen_model.endswith("(user)"):
            base_name = chosen_model.replace(" (user)", "")
            local_path = os.path.join(self.model_dir, base_name + ".pt")
            print(f"Загружаем пользовательскую модель: {local_path}")
            self.model = YOLO(local_path)
        else:
            # Преобразуем "YOLOv8n" -> "yolov8n.pt"
            pt_name = f"{chosen_model.lower()}.pt"
            print(f"Загружаем стандартную модель: {pt_name}")
            self.model = YOLO(pt_name)

        # Обновляем список имен классов
        self.class_names = list(self.model.names.values()) if self.model else []

    def inference(self):
        """
        Основной метод:
          1) Отображение UI.
          2) Считывание настроек из боковой панели.
          3) Запуск видеопотока с выбранного источника камеры.
        """
        self.web_ui()
        start_btn = self.sidebar()

        if start_btn:
            cap = cv2.VideoCapture(self.camera_source)
            if not cap.isOpened():
                self.st.error("Не удалось открыть камеру. Проверьте настройки.")
                return

            stop_button = self.st.button("Остановить")

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    self.st.warning("Не удалось считать кадр. Остановка.")
                    break

                # Если включён трекинг, используем метод track, иначе обычная детекция
                if self.enable_trk == "Yes":
                    results = self.model.track(
                        frame,
                        conf=self.conf,
                        iou=self.iou,
                        classes=self.selected_indices,
                        persist=True
                    )
                else:
                    results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_indices)

                # Получаем детекции и собираем найденные классы
                boxes = results[0].boxes
                detected_classes = set()
                for det in boxes:
                    cls_id = int(det.cls[0])
                    cls_name = self.model.names[cls_id]
                    detected_classes.add(cls_name)

                if detected_classes:
                    classes_str = ", ".join(detected_classes)
                    self.st.toast(f"Обнаружены классы: {classes_str}")

                # Генерируем аннотированный кадр
                annotated_frame = results[0].plot()

                if stop_button:
                    cap.release()
                    self.st.stop()

                # Отображаем оригинальный и аннотированный кадры
                self.org_frame.image(frame, channels="BGR", caption="Оригинал")
                self.ann_frame.image(annotated_frame, channels="BGR", caption="С детекцией")

            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = Inference()
    app.inference()
