import argparse
import glob
import os
import warnings

import pandas as pd

from src.data_preparation import DataPreparation
from src.data_splitting import DataSplitting
from src.data_visualization import DataVisualization
from src.feature_engineering import FeatureEngineering
from src.model_prediction import ModelPrediction
from src.model_training import ModelTraining
from src.postgres import PostgresManager
from src.setup import Setup

warnings.filterwarnings("ignore")


class AudioMNIST:
    """
    This class is used to run the main pipeline
    """

    def __init__(self):
        self.config_file = SU.read_config()
        self.pgs_file = PM.read_config()

    def DataPrepare(self):
        meta_data = SU.read_file(SU.set_audio_path(), self.config_file["meta_data"])

        df = pd.DataFrame()

        num_folders = len(next(os.walk(SU.set_audio_path()))[1]) + 1

        for i in range(1, num_folders):
            SU.loop_progress(i, num_folders - 1, 6)

            audio_file = sorted(
                glob.glob(os.path.join(SU.set_audio_path(), f"{i:02d}", "*.wav"))
            )

            for file in audio_file:
                dig, vp, rep = SU.extract_file_info(file)
                fs, audio_data = DP.read_audio(file)
                audio_data = DP.resample_data(fs, audio_data)
                audio_data = DP.zero_pad(audio_data)
                time_domain_features = DP.feature_creation_time_domain(audio_data)
                fft_data = DP.fft_data(audio_data)
                features = DP.feature_creation_frequency_domain(fft_data)
                features.update(time_domain_features)
                n_features = DP.normalize_features(features)
                features = DP.add_column_dict(
                    n_features,
                    self.config_file["target"],
                    meta_data[vp][self.config_file["target"]],
                )
                df = pd.concat(
                    [df, pd.DataFrame(features, index=[0])], ignore_index=True
                )

                DV.plot_audio(
                    fs,
                    audio_data,
                    self.config_file["plots"]["audio"].format(dig[-1], vp, rep),
                    1,
                )
                DV.plot_stft(
                    fs,
                    audio_data,
                    self.config_file["plots"]["stft"].format(dig[-1], vp, rep),
                    1,
                )
                DV.play_audio(file, 0)

        df.to_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["prepared"]),
            index=False,
        )

        gender_count = DP.column_value_counts(df, self.config_file["target"])
        print(f"Female audio recordings: {gender_count[0]}")
        print(f"Male audio recordings: {gender_count[1]}")
        print(f"Prepared dataset, columns: {df.shape[1]} and rows: {df.shape[0]}")

    def FeatureEngineer(self):
        df = pd.read_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["prepared"])
        )
        df = FE.remove_constant_columns(df, self.config_file["target"])
        df = FE.categorize_column_values(df, self.config_file["target"])

        DV.plot_column_dist(
            df,
            self.config_file["plots"]["column_distribution"],
            self.config_file["target"],
        )

        corr_matrix = FE.pearson_correlation(df, self.config_file["target"])

        DV.plot_corr_matrix(
            corr_matrix, self.config_file["plots"]["correlation_matrix"]
        )

        df = FE.remove_correlated_columns(
            df,
            self.config_file["thresholds"]["correlation"],
            self.config_file["target"],
        )

        df.to_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["engineered"]),
            index=False,
        )

    def DataSplit(self):
        df = pd.read_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["engineered"])
        )

        self.train_df, self.val_df, self.test_df = DS.split(
            df,
            self.config_file["target"],
            self.config_file["datasplit"]["val_size"],
            self.config_file["datasplit"]["test_size"],
        )

        print(
            f"Training set, columns: {self.train_df.shape[1]} and rows: {self.train_df.shape[0]}"
        )
        print(
            f"Validation set, columns: {self.val_df.shape[1]} and rows: {self.val_df.shape[0]}"
        )
        print(
            f"Test set, columns: {self.test_df.shape[1]} and rows: {self.test_df.shape[0]}"
        )

        (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        ) = DS.prepare_data(
            self.train_df, self.val_df, self.test_df, self.config_file["target"]
        )

    def ModelTune(self):
        MT.set_params_grid(
            SU.read_file(
                SU.set_model_path(),
                self.config_file["parameters"]["model_hyperparameters"],
            )
        )
        MT.grid_search(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
        )
        MT.save_best_params(
            SU.set_result_path(), self.config_file["results"]["model_best_params"]
        )

    def ModelTrain(self):
        MT.set_params_fit(
            SU.read_file(
                SU.set_model_path(), self.config_file["parameters"]["model_parameters"]
            )
        )
        MT.fit(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
        )

        MT.save_model_object(
            SU.set_result_path(),
            self.config_file["results"]["model_object"],
        )

        MT.save_eval_metrics(
            SU.set_result_path(),
            self.config_file["results"]["model_results"],
        )

        df = MT.create_log_df(
            SU.read_file(
                SU.set_result_path(), self.config_file["results"]["model_results"]
            )
        )

        DV.plot_feature_importance(
            MT.model,
            self.config_file["plots"]["feature_importance"],
        )
        DV.plot_loss(
            df["iteration"],
            df["train_loss"],
            df["val_loss"],
            self.config_file["plots"]["model_loss"],
        )
        DV.plot_accuracy(
            df["iteration"],
            df["train_acc"],
            df["val_acc"],
            self.config_file["plots"]["model_accuracy"],
        )

    def ModelPredict(self):
        MT.model = MP.load_model(
            SU.set_result_path(), self.config_file["results"]["model_object"]
        )

        y_pred = MP.predict(MT.model, self.X_test)

        df = self.test_df.reset_index(drop=True)

        df = DP.add_column_df(df, self.config_file["predicted"], y_pred)

        df.to_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["predicted"]),
            index=False,
        )

        DV.plot_confusion_matrix(
            self.y_test,
            y_pred,
            self.config_file["labels"],
            self.config_file["plots"]["confusion_matrix"],
        )
        DV.plot_shapley_summary(
            MT.model,
            self.X_test,
            self.config_file["plots"]["shapley_summary"],
        )

        MP.evaluate_predictions(self.y_test, y_pred)

    def DataPostgres(self):
        PM.drop_table(self.pgs_file["table"]["prepared"])
        PM.drop_table(self.pgs_file["table"]["engineered"])
        PM.drop_table(self.pgs_file["table"]["predicted"])

        PM.create_table_from_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["prepared"]),
            self.pgs_file["table"]["prepared"],
            self.config_file["target"],
        )
        PM.create_table_from_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["engineered"]),
            self.pgs_file["table"]["engineered"],
        )
        PM.create_table_from_csv(
            os.path.join(SU.set_data_path(), self.config_file["data"]["predicted"]),
            self.pgs_file["table"]["predicted"],
        )

        PM.write_csv_to_table(
            os.path.join(SU.set_data_path(), self.config_file["data"]["prepared"]),
            self.pgs_file["table"]["prepared"],
        )
        PM.write_csv_to_table(
            os.path.join(SU.set_data_path(), self.config_file["data"]["engineered"]),
            self.pgs_file["table"]["engineered"],
        )
        PM.write_csv_to_table(
            os.path.join(SU.set_data_path(), self.config_file["data"]["predicted"]),
            self.pgs_file["table"]["predicted"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--cfg_file",
        type=str,
        default="config.yaml",
        help="Configuration File",
    )
    parser.add_argument(
        "-y",
        "--pgs_file",
        type=str,
        default="postgres.yaml",
        help="Postgres File",
    )
    parser.add_argument(
        "-d",
        "--data_prep",
        type=str,
        default="true",
        help="Data Preparation",
    )
    parser.add_argument(
        "-f",
        "--feat_eng",
        type=str,
        default="true",
        help="Feature Engineering",
    )
    parser.add_argument(
        "-s",
        "--data_split",
        type=str,
        default="true",
        help="Data Splitting",
    )
    parser.add_argument(
        "-u",
        "--model_tune",
        type=str,
        default="true",
        help="Model Tuning",
    )
    parser.add_argument(
        "-t",
        "--model_train",
        type=str,
        default="true",
        help="Model Training",
    )
    parser.add_argument(
        "-p",
        "--model_pred",
        type=str,
        default="true",
        help="Model Prediction",
    )
    parser.add_argument(
        "-q",
        "--data_sql",
        type=str,
        default="false",
        help="Data To PostgreSQL",
    )
    args = parser.parse_args()

    SU = Setup(args.cfg_file)
    PM = PostgresManager(args.pgs_file)
    DP = DataPreparation()
    DV = DataVisualization(SU.set_plot_path())
    FE = FeatureEngineering()
    DS = DataSplitting()
    MT = ModelTraining()
    MP = ModelPrediction()
    AM = AudioMNIST()

    data_split_required = (
        args.model_train == "true"
        or args.model_tune == "true"
        or args.model_pred == "true"
    )

    if args.data_prep == "true":
        AM.DataPrepare()

    if args.feat_eng == "true":
        AM.FeatureEngineer()

    if args.data_split == "true" or data_split_required:
        AM.DataSplit()

    if args.model_tune == "true":
        AM.ModelTune()

    if args.model_train == "true":
        AM.ModelTrain()

    if args.model_pred == "true":
        AM.ModelPredict()

    if args.data_sql == "true":
        AM.DataPostgres()
