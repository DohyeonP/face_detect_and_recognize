import os

# parsing 을 위한 import
import argparse

class main_process():
    def __init__(self):
        # 현재 디렉토리 경로
        self.current_path = os.path.dirname(os.path.abspath(__file__))

    # 웹캠으로부터 학습할 이미지 가져오기
    def get_train_image(self, name):
        import Get_Image_From_Webcam

        # image 객체 생성
        image = Get_Image_From_Webcam.get_image_from_webcam(name)

        # 웹캠 실행 및 이미지 저장
        image.get_image()

    # 학습할 이미지에서 랜덤하게 20가지의 테스트 이미지 가져오기
    def get_test_image(self, num_images=20):
        import random
        import shutil
        from pathlib import Path

        # 학습할 이미지를 가져올 디렉토리
        source_directory = os.path.join(self.current_path, 'train_photos')

        # 테스트 이미지를 저장할 디렉토리
        dest_directory = os.path.join(self.current_path, 'test_photos')

        # source_directory에서 하위 디렉토리 목록 가져오기
        subdirectories = [d for d in Path(source_directory).iterdir() if d.is_dir()]

        for subdir in subdirectories:
            subdir_name = subdir.name
            dest_subdir = Path(dest_directory) / subdir_name

            # 목적지 디렉토리 생성
            dest_subdir.mkdir(parents=True, exist_ok=True)

            # 이미지를 목적지 디렉토리로 복사
            image_files = [f for f in subdir.glob('*.jpg') if f.is_file()]
            selected_images = random.sample(image_files, min(num_images, len(image_files)))

            for image_path in selected_images:
                dest_path = dest_subdir / image_path.name
                shutil.copy(image_path, dest_path)

    # 학습할 데이터와 테스트할 데이터를 구축
    def set_data(self):
        import Data_Setting

        # data 객체 생성
        data = Data_Setting.data_setting()

        # 학습할 데이터와 테스트할 데이터 압축
        data.compressing_all_dataset()

    # 구축한 데이터를 가지고 embedding 하기
    def set_embeddings_data(self):
        import Get_Embeddings_From_Data_Set

        obj = Get_Embeddings_From_Data_Set.Embeddings()
        obj.process_and_save_embeddings()

    # embedding 한 데이터를 가지고 얼굴 분류하기
    def face_classification_using_embedding_data(self):
        import Face_Classification

        obj = Face_Classification.face_classification()
        obj.run_face_classification()

    def display_parser_help(self, parser_name):
        # 사용 가능한 파서 리스트
        available_parsers = {
            "train_image_name": self.get_train_image,
            "test_images": self.get_test_image,
            "set_data": self.set_data,
            "set_embeddings": self.set_embeddings_data,
            "face_classification": self.face_classification_using_embedding_data,
        }

        # 사용 가능한 파서 리스트 출력
        if parser_name == "all":
            print("사용가능한 parsers:")
            for parser in available_parsers:
                print(f"  --{parser}")
            return

        # 특정 파서에 대한 도움말 출력
        if parser_name in available_parsers:
            print(f"Help for --{parser_name}:")
            print(f"{available_parsers[parser_name].__doc__}")
        else:
            print(f"Parser --{parser_name} not found.")

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="Main process for face-related tasks.")
        parser.add_argument("--train_image_name", type=str, help="Name for training image")
        parser.add_argument("--test_images", type=int, default=20, help="Number of test images")
        parser.add_argument("--set_data", action="store_true", help="Set up training and testing data")
        parser.add_argument("--set_embeddings", action="store_true", help="Set up embeddings data")
        parser.add_argument("--face_classification", action="store_true", help="Run face classification using embeddings data")
        parser.add_argument("--help_parser", type=str, default="all", help="Display help for a specific parser")
        return parser.parse_args()

    def run_tasks(self):
        args = self.parse_arguments()

        # 인자 출력 추가
        print(args)

        # 도움말을 출력하는 파서가 지정되었을 경우 도움말 출력 후 종료
        if args.help_parser:
            self.display_parser_help(args.help_parser)
            return

        # 이름을 지정하고 웹캠으로부터 얼굴 이미지를 받아온다.
        # 해당 이미지의 이름은 지정된 이름으로 저장된다.
        if args.train_image_name:
            self.get_train_image(args.train_image_name)

        # 저장된 이미지들 중에서 랜덤으로 테스트할 이미지를 추출한다.
        # 이 작업은 사전에 학습할 이미지가 있어야 가능하다.
        # 추출할 이미지 갯수는 기본이 20장이고
        # 만약 명령어로 특정한 숫자를 입력하면 해당 숫자만큼 테스트 이미지가 추출된다.
        if args.test_images:
            self.get_test_image(args.test_images)

        # 저장된 학습할 이미지와 테스트 이미지를 학습 모델에 입력할 수 있도록 구축한다.
        # numpy 배열로 이미지에서 데이터를 추출하고 해당 이미지의 이름으로 라벨링한다.
        if args.set_data:
            self.set_data()

        # 구축된 데이터를 임베딩한다.
        # 이 작업은 사전에 데이터를 구축해야만 가능하다.
        if args.set_embeddings:
            self.set_embeddings_data()

        # 임베딩된 데이터를 가지고 얼굴을 뷴류한다.
        if args.face_classification:
            self.face_classification_using_embedding_data()


if __name__ == "__main__":
    run_main = main_process()
    run_main.run_tasks()
