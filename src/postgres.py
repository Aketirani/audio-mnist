import csv
import os

import psycopg2
import yaml


class PostgresManager:
    """
    This class provides methods to manage PostgreSQL database connections and operations
    """

    def __init__(self, pgs_file: str) -> None:
        """
        Initialize the class

        :param pgs_file: str, path to the PostgreSQL configuration file
        """
        self.pgs_file = pgs_file

    def read_config(self) -> dict:
        """
        Read the config yaml file and return the data as a dictionary

        :return: dict, containing the configuration data
        """
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_folder = os.path.join(script_dir, "../config")
            os.chdir(config_folder)
            with open(self.pgs_file, "r") as file:
                pgs_setup = yaml.safe_load(file)
        except:
            raise FileNotFoundError(f"{self.pgs_file} is not a valid config filepath!")
        return pgs_setup

    def _connect_to_database(self) -> None:
        """
        Connect to the PostgreSQL database
        """
        config = self.read_config()["connection"]
        try:
            self.conn = psycopg2.connect(**config)
        except psycopg2.Error as e:
            raise e

    def _execute_query(self, query: str) -> None:
        """
        Execute a SQL query on the connected PostgreSQL database

        :param query: str, SQL query to be executed
        """
        try:
            self._connect_to_database()
            cur = self.conn.cursor()
            cur.execute(query)
            self.conn.commit()
        except psycopg2.Error as e:
            self.conn.rollback()
            raise e
        finally:
            cur.close()
            self.conn.close()

    def write_csv_to_table(self, file_path: str, table_name: str) -> None:
        """
        Load data from a CSV file into a PostgreSQL table using the COPY command

        :param file_path: str, path to the CSV file containing data
        :param table_name: str, name of the PostgreSQL table to write data into
        """
        try:
            query = f"COPY {table_name} FROM '{file_path}' DELIMITER ',' CSV HEADER;"
            self._execute_query(query)
        except psycopg2.Error as e:
            self.conn.rollback()
            raise e

    def create_table_from_csv(
        self, file_path: str, table_name: str, target_column: str = None
    ) -> None:
        """
        Create a table in the PostgreSQL database based on the columns in a CSV file

        :param file_path: str, path to the CSV file containing column names and sample data
        :param table_name: str, name of the table to create
        :param target_column: str, name of the target variable column (default is None)
        """
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            columns = next(reader)
            column_definitions = [
                f"{column} {'VARCHAR(10)' if column == target_column else 'NUMERIC'}"
                for column in columns
            ]
            column_definitions = ", ".join(column_definitions)
        query = f"CREATE TABLE {table_name} ({column_definitions});"
        self._execute_query(query)

    def drop_table(self, table_name: str) -> None:
        """
        Drop a table from the PostgreSQL database

        :param table_name: str, name of the table to drop
        """
        query = f"DROP TABLE IF EXISTS {table_name};"
        self._execute_query(query)
