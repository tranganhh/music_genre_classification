import argparse
import sys
import pickle


from train.train import (
    generate_knn_classifier,
    train,
)

from predict.predict import (
    evaluation,
)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--folder_path")
    args = parser.parse_args()

    if args.mode == "train":
        print("train")
        features = r'data\features_30_sec.csv'
        file_path = r'C:\Users\trang\Desktop\Exemple_projet\music_classification'
        classifier = generate_knn_classifier()
        classifier = train('{}\%s'.format(file_path)%(features), classifier=classifier)

        with open('model_classification.pkl', 'wb') as f:
            pickle.dump(classifier, f)
        


    elif args.mode == "predict":
        print("predict")
        files = r'data\features_30_sec.csv'
        file_path = r'C:\Users\trang\Desktop\Exemple_projet\music_classification'
        
        #for file in files : 
        res = evaluation(filecsv='{}\%s'.format(file_path)%(files))


    return 0


if __name__ == "__main__":
    sys.exit(main())
