import pickle
import time
import codecs
import pandas
import pandas as pd
from PdmContext.utils.structure import Context



class SQLiteHandler:

    def __init__(self, db_name):
        """

        **db_name**: the name of the database to load or create
        """
        import sqlite3
        self.conn = sqlite3.connect(db_name, isolation_level='DEFERRED')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''PRAGMA synchronous = OFF''')
        self.cursor.execute('''PRAGMA journal_mode = OFF''')

        self.create_table()
        # self.create_index(self, field_name)

    def create_table(self):
        """
        Create a table with fields Date (datetime) and 3 text fields (target, contextpickle, metadata,value)
        target is the name of the target values,
        context pickle contain a Context object in binary form after pickle.dump()
        metadata, contain users metadate
        value, contain the last value of target series.

        """
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS my_table (
                Date INTEGER,
                target TEXT,
                contextpickle TEXT,
                metadata TEXT,
                value REAL,
                PRIMARY KEY (Date, target)
            )
        ''')
        self.conn.commit()

    def create_index(self, field_name):
        # Create an index on a specified field (e.g., target)
        self.cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{field_name} ON my_table({field_name})')
        self.conn.commit()

    def insert_record(self, date: pandas.Timestamp, target, context: Context, metadata=""):
        """
        Inserts a record of DATE,TARGET,Context (after pickle) , Meta data text, Value (which is the last target value from context)

        **Parameters**:

        **date**: The timestamp of the context

        **target**: The target name on which the context was built

        **context**: The Context object

        **metadata**: Text of possible meta data.
        """

        unix_timestamp = int(time.mktime(date.timetuple()))
        tosave = self.create_pickle(context)
        self.cursor.execute(
            'INSERT OR REPLACE INTO my_table (Date, target, contextpickle,metadata,value) VALUES (?, ?, ?, ?, ?)',
            (unix_timestamp, str(target), tosave, metadata, context.CD[target][-1]))
        self.conn.commit()

    def get_records_between_dates(self, start_date, end_date, target):
        # Convert the start and end dates to Unix timestamps
        start_timestamp = int(time.mktime(start_date.timetuple()))
        end_timestamp = int(time.mktime(end_date.timetuple()))
        # Retrieve records between the specified dates
        self.cursor.execute('SELECT * FROM my_table WHERE Date >= ? AND Date <= ? AND target = ?',
                            (start_timestamp, end_timestamp, target))
        records = self.cursor.fetchall()
        return records

    def get_contex_between_dates(self, start_date: pd.Timestamp, end_date: pd.Timestamp, target):
        """
           Returns all Context object in list of the specified target between the start and end dates given.

           **Parameter**:

           **start_date**: Begin date of the query objects

           **end_date**: End date of the query objects

           **target**: the target of which the context will be returned

           **return**: A list of from PdmContext.utils.structure.Context objects
        """
        # Convert the start and end dates to Unix timestamps
        start_timestamp = int(time.mktime(start_date.timetuple()))
        end_timestamp = int(time.mktime(end_date.timetuple()))

        # Retrieve records between the specified dates
        self.cursor.execute('SELECT * FROM my_table WHERE Date >= ? AND Date <= ? AND target = ?',
                            (start_timestamp, end_timestamp, target))
        records = self.cursor.fetchall()
        return_list = []
        for record in records:
            date, target, context, metadata, value = record
            # If you stored the Date as Unix timestamp, convert it to a datetime object
            contextdict = self.load_pickle(context)
            return_list.append(contextdict)
        return return_list

    def get_all_context_by_target(self, target):
        """
           Returns all Context object in list of the specified target.

           **Parameter**:

           **target**: the target of which the context will be returned

           **return**: A list of from PdmContext.utils.structure.Context objects
        """
        # Convert the start and end dates to Unix timestamps

        # Retrieve records between the specified dates
        self.cursor.execute('SELECT * FROM my_table WHERE target = ?', (target,))
        records = self.cursor.fetchall()
        return_list = []
        for record in records:
            date, target, context, metadata, value = record
            # If you stored the Date as Unix timestamp, convert it to a datetime object
            contextdict = self.load_pickle(context)
            return_list.append(contextdict)
        return return_list

    def load_pickle(self, data_string):

        # Convert the string back to bytes
        dcccoded = codecs.decode(data_string.encode(), "base64")

        loaded_data = pickle.loads(dcccoded)

        return loaded_data

    def create_pickle(self, data):
        # Serialize the data to a bytes object
        data_bytes = codecs.encode(pickle.dumps(data), "base64").decode()

        # Convert the bytes to a string
        data_string = str(data_bytes)

        return data_string

    def get_records_by_index(self, field_name, value):
        # Retrieve records based on the indexed field (e.g., target)
        self.cursor.execute(f'SELECT * FROM my_table WHERE {field_name} = ?', (value,))
        records = self.cursor.fetchall()
        return records

    def get_records_by_fields(self, field_name1, field1_value, field_name2, field2_value):
        # Retrieve records based on the values of Field1 and Field2
        self.cursor.execute(f'SELECT * FROM my_table WHERE {field_name1} = ? AND {field_name2} = ?',
                            (field1_value, field2_value))
        records = self.cursor.fetchall()
        return records

    def close_connection(self):
        # Close the database connection
        self.conn.close()


class InfluxDBHandler:

    def __init__(self, host='localhost', port=8086, db_name='my_database', measurment_name="my_table"):
        """

        **host**: location of host (ip)

        **port**: port where the database is hosted

        **db_name**: Database name

        **measurment_name**:  measurment name to be used
        """
        from influxdb import InfluxDBClient
        self.client = InfluxDBClient(host, port, db_name)
        self.client.create_database(db_name)
        self.db_name = db_name
        self.my_measurment = measurment_name

    def create_table(self):
        pass  # In InfluxDB, you don't need to explicitly create a table or measurement; it will be created on data insertion.

    def create_index(self, name):
        pass  # In InfluxDB, you don't need to explicitly create a table or measurement; it will be created on data insertion.

    def insert_record(self, date: pd.Timestamp, target, context: Context, meta_data=""):
        """
        Create a measurment with fields Date (datetime) and 3 text fields (target, contex (pickle), metadata,value)
        target is the name of the target values,
        context pickle contain a Context object in binary form after pickle.dump()
        metadata, contain users metadate
        value, contain the last value of target series.

        **Parameters**:

        **date**: Date of the context

        **target**: name of target series (for tag field)

        **context**: context object

        **meta_data**: meta data passed by user

        """
        unix_timestamp = int(time.mktime(date.timetuple()))
        contextpickle = self.create_pickle(context)

        data = [
            {
                "measurement": f"{self.my_measurment}",
                "tags": {
                    "target": target,
                },
                "time": unix_timestamp * 10 ** 9,  # InfluxDB uses nanoseconds for timestamp
                "fields": {
                    "contextpickle": contextpickle,
                    "metadata": meta_data,
                    "value": context.CD[target][-1]
                }
            }
        ]

        self.client.write_points(data, database=self.db_name)

    def get_contex_between_dates(self, start_date: pd.Timestamp, end_date: pd.Timestamp, target):
        """
           Returns all Context object in list of the specified target between the start and end dates given.

           **Parameter**:

           **start_date**: Begin date of the query objects

           **end_date**: End date of the query objects

           **target**: the target of which the context will be returned (tag)

           **return**: A list of from PdmContext.utils.structure.Context objects
        """

        query = f'SELECT * FROM "{self.my_measurment}" WHERE ("target"::tag = \'{target}\') AND time >= {start_date.timestamp()} AND time <= {end_date.timestamp()}'
        result = self.client.query(query, database=self.db_name)
        return_list = []

        for record in result.get_points():
            contextdict = self.load_pickle(record['contextpickle'])
            return_list.append(contextdict)

        return return_list

    def get_all_context_by_target(self, target):
        """
        Returns all Context object in list of the specified target.

        **Parameter**:

        **target**: the target of which the context will be returned (tag)

        **return**: A list of from PdmContext.utils.structure.Context objects
        """
        query = (f'SELECT * FROM "{self.my_measurment}" WHERE ("target"::tag = \'{target}\') ')
        result = self.client.query(query, database=self.db_name)
        return_list = []

        for record in result.get_points():
            contextdict = self.load_pickle(record[f'contextpickle'])
            return_list.append(contextdict)

        return return_list

    def load_pickle(self, data_string):
        # Convert the string back to bytes
        data_string = data_string.split("\\n")[:-1]
        original = ""
        for sstring in data_string:
            original += f"{sstring}\n"
        encoded = original.encode()
        dcccoded = codecs.decode(encoded, "base64")

        loaded_data = pickle.loads(dcccoded)

        return loaded_data

    def create_pickle(self, data):
        # Serialize the data to a bytes object
        data_bytes = codecs.encode(pickle.dumps(data), "base64").decode()

        # Convert the bytes to a string
        data_string = str(data_bytes)

        return data_string

    def close_connection(self):
        pass  # InfluxDB connection is closed automatically when the client is closed
