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
            # change the directory to the configuration folder
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_folder = os.path.join(script_dir, "../config")
            os.chdir(config_folder)

            # open the configuration folder
            with open(self.pgs_file, "r") as file:
                # load the configuration file into a dictionary
                pgs_setup = yaml.safe_load(file)
        except:
            # raise an error if the filename is not valid
            raise FileNotFoundError(f"{self.pgs_file} is not a valid config filepath!")

        # return the configuration data
        return pgs_setup

    def _connect_to_database(self) -> None:
        """
        Connect to the PostgreSQL database
        """
        # read config file
        config = self.read_config()["connection"]
        try:
            # connect to the database
            self.conn = psycopg2.connect(**config)
        except psycopg2.Error as e:
            # raise error
            raise e

    def _execute_query(self, query: str) -> None:
        """
        Execute a SQL query on the connected PostgreSQL database

        :param query: str, SQL query to be executed
        """
        try:
            # connect to the database
            self._connect_to_database()
            # create a cursor
            cur = self.conn.cursor()
            # execute the query
            cur.execute(query)
            # commit
            self.conn.commit()
        except psycopg2.Error as e:
            # rollback and raise error
            self.conn.rollback()
            raise e
        finally:
            # close the cursor and connection
            cur.close()
            self.conn.close()

    def write_csv_to_table(self, file_path: str, table_name: str) -> None:
        """
        Load data from a CSV file into a PostgreSQL table using the COPY command

        :param file_path: str, path to the CSV file containing data
        :param table_name: str, name of the PostgreSQL table to write data into
        """
        try:
            # construct the query
            query = f"COPY {table_name} FROM '{file_path}' DELIMITER ',' CSV HEADER;"
            # execute the COPY command with the file path as a parameter
            self._execute_query(query)
        except psycopg2.Error as e:
            # rollback and raise error
            self.conn.rollback()
            raise e

    def create_table_from_csv(self, file_path: str, table_name: str) -> None:
        """
        Create a table in the PostgreSQL database based on the columns in a CSV file

        :param file_path: str, path to the CSV file containing column names and sample data
        :param table_name: str, name of the table to create
        """
        # read the CSV file to get column names and sample data types
        with open(file_path, "r") as file:
            # use the first row to get column names
            reader = csv.reader(file)
            columns = next(reader)
            # assume all columns are of type NUMERIC for simplicity
            column_definitions = ", ".join([f"{column} NUMERIC" for column in columns])

        # construct the query
        query = f"CREATE TABLE {table_name} ({column_definitions});"

        # execute the query
        self._execute_query(query)

    def drop_table(self, table_name: str) -> None:
        """
        Drop a table from the PostgreSQL database

        :param table_name: str, name of the table to drop
        """
        # construct the query
        query = f"DROP TABLE IF EXISTS {table_name};"

        # execute the query
        self._execute_query(query)
