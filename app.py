import io
import os
import cv2

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS


class Inference:
    def __init__(self):
        check_requirements("streamlit>=1.29.0")
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

        # Итоговый список выбора
        self.model_options = self.available_models + self.user_models

        # Поля, которые будем наполнять/изменять
        self.model = None
        self.class_names = []
        self.selected_classes = []
        self.selected_indices = []

        self.source = None       # Источник видео (camera / video)
        self.conf = 0.50         # Confidence Threshold
        self.iou = 0.50          # IoU Threshold
        self.enable_trk = False  # Флаг включения трекинга
        self.vid_file_name = None

        # UI элементы для отображения
        self.org_frame = None
        self.ann_frame = None

        # Храним имя уже загруженной модели, чтобы не загружать повторно
        self.loaded_model_name = None

    def _list_user_models(self):
        """
        Собирает список всех .pt-файлов из папки self.model_dir
        и возвращает их, добавляя пометку " (user)" к имени.
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
        Простейшее оформление титульной части.
        """
        self.st.set_page_config(page_title="YOLO App", layout="wide")
        title_html = """<div>
            <h2 style="color:#FF64DA; text-align:center; margin-bottom:10px;">YOLO Detection App</h2>
        </div>"""
        self.st.markdown(title_html, unsafe_allow_html=True)

    def sidebar(self):
        """
        Боковая панель с настройками:
         1) Загрузка пользовательской модели
         2) Выбор модели
         3) Параметры детекции (Confidence, IoU, Tracking)
         4) Выбор источника (camera / video)
         5) Выбор классов
         6) Кнопка "Старт"
        """
        self.st.sidebar.title("Настройки")

        # (1) Загрузка пользовательской модели
        model_file = self.st.sidebar.file_uploader("Загрузите модель (.pt)", type=["pt"])
        if model_file is not None:
            saved_path = os.path.join(self.model_dir, model_file.name)
            with open(saved_path, "wb") as f:
                f.write(model_file.read())
            self.st.sidebar.success(f"Модель '{model_file.name}' загружена!")
            # Обновим список пользовательских моделей
            self.user_models = self._list_user_models()
            self.model_options = self.available_models + self.user_models

        # (2) Выбор модели
        chosen_model = self.st.sidebar.selectbox("Модель YOLO", self.model_options)

        # (3) Параметры детекции: Confidence, IoU, Tracking
        self.conf = self.st.sidebar.slider("Порог достоверности", 0.0, 1.0, self.conf, 0.01)
        self.iou = self.st.sidebar.slider("Порог пересечения", 0.0, 1.0, self.iou, 0.01)
        self.enable_trk = self.st.sidebar.radio("Включение трекинга", ("Yes", "No"))

        # (4) Выбор источника
        self.source = self.st.sidebar.selectbox("Выберите источник:", ("camera", "video"))

        # Если модель изменилась - загрузим её и вытащим список классов
        if chosen_model != self.loaded_model_name:
            self._load_model(chosen_model)
            self.loaded_model_name = chosen_model

        # (5) Если модель загружена, выводим выбор классов
        if self.model is not None:
            default_classes = self.class_names[:3] if len(self.class_names) >= 3 else self.class_names
            self.selected_classes = self.st.sidebar.multiselect(
                "Выберите классы", self.class_names, default=default_classes
            )
            self.selected_indices = [self.class_names.index(c) for c in self.selected_classes]

        col1, col2 = self.st.columns(2)
        self.org_frame = col1.empty()
        self.ann_frame = col2.empty()

        # (6) Кнопка "Старт" для запуска инференса
        start_btn = self.st.sidebar.button("Старт")
        return start_btn

    def _load_model(self, chosen_model):
        """
        Загрузка модели:
         - Если выбрана (user), берём файл из models/<имя>.pt
         - Иначе скачиваем/используем стандартную модель (yolov8n.pt и т. д.)
        После загрузки обновляем self.class_names.
        """
        if chosen_model.endswith("(user)"):
            base_name = chosen_model.replace(" (user)", "")
            local_path = os.path.join(self.model_dir, base_name + ".pt")
            LOGGER.info(f"Загружаем пользовательскую модель: {local_path}")
            self.model = YOLO(local_path)
        else:
            # Преобразуем "YOLOv8n" -> "yolov8n.pt"
            pt_name = f"{chosen_model.lower()}.pt"
            LOGGER.info(f"Загружаем стандартную модель: {pt_name}")
            self.model = YOLO(pt_name)

        # Получим список имён классов
        self.class_names = list(self.model.names.values()) if self.model else []

    def inference(self):
        """
        Основной метод:
          1) Рисуем UI
          2) Считываем настройки боковой панели
          3) Если нажата кнопка "Старт", начинаем обработку видео/камеры
        """
        self.web_ui()
        start_btn = self.sidebar()

        if start_btn:
            # Определяем источник (камера/файл)
            if self.source == "camera":
                self.vid_file_name = 0
            else:
                # Выбор и загрузка видео
                vid_file = self.st.sidebar.file_uploader("Загрузите видео", type=["mp4","mov","avi","mkv"])
                if vid_file is None:
                    self.st.warning("Не выбрано видео для загрузки.")
                    return
                g = io.BytesIO(vid_file.read())
                with open("uploaded_video.mp4", "wb") as out:
                    out.write(g.read())
                self.vid_file_name = "uploaded_video.mp4"

            cap = cv2.VideoCapture(self.vid_file_name)
            if not cap.isOpened():
                self.st.error("Не удалось открыть камеру или файл. Проверьте настройки.")
                return

            stop_button = self.st.button("Остановить")

            # while cap.isOpened():
            #     success, frame = cap.read()
            #     if not success:
            #         self.st.warning("Не удалось считать кадр. Остановка.")
            #         break

            #     # Детекция или трекинг в зависимости от enable_trk
            #     if self.enable_trk == "Yes":
            #         results = self.model.track(
            #             frame,
            #             conf=self.conf,
            #             iou=self.iou,
            #             classes=self.selected_indices,
            #             persist=True
            #         )
            #     else:
            #         results = self.model(
            #             frame,
            #             conf=self.conf,
            #             iou=self.iou,
            #             classes=self.selected_indices
            #         )

            #     annotated_frame = results[0].plot()

            #     if stop_button:
            #         cap.release()
            #         self.st.stop()

            #     self.org_frame.image(frame, channels="BGR", caption="Оригинал")
            #     self.ann_frame.image(annotated_frame, channels="BGR", caption="С детекцией")

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    self.st.warning("Не удалось считать кадр. Остановка.")
                    break

                # Если трекинг включён
                if self.enable_trk == "Yes":
                    results = self.model.track(
                        frame,
                        conf=self.conf,
                        iou=self.iou,
                        classes=self.selected_indices,
                        persist=True  # для трекинга обычно указывают persist=True
                    )
                else:
                    # Обычная детекция без трекинга
                    results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_indices)

                # Берём предсказания из первого (и обычно единственного) результата
                boxes = results[0].boxes  # список детекций (bounding boxes)

                # Собираем названия классов, которые модель нашла
                detected_classes = set()
                for det in boxes:
                    cls_id = int(det.cls[0])       # ID класса (например, 0, 1, 2 ...)
                    cls_name = self.model.names[cls_id]  # Имя класса по ID
                    detected_classes.add(cls_name)

                # Если какие-то классы обнаружены, выводим всплывающий toast
                if detected_classes:
                    classes_str = ", ".join(detected_classes)
                    self.st.toast(f"Обнаружены классы: {classes_str}")

                # Генерируем кадр с аннотациями
                annotated_frame = results[0].plot()

                # Проверяем, не нажата ли кнопка "Остановить"
                if stop_button:
                    cap.release()
                    self.st.stop()

                # Отображаем исходный и аннотированный кадры
                self.org_frame.image(frame, channels="BGR", caption="Оригинал")
                self.ann_frame.image(annotated_frame, channels="BGR", caption="С детекцией")


            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = Inference()
    app.inference()
