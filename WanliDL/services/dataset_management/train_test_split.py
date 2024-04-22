import uuid
from datetime import datetime
import os
import random
import shutil
from . import DatasetManagementService

class TrainTestSplitVisionDatasetManagementService(DatasetManagementService):
    def ensure_table_exists(self):
        """Create the table for storing train/test split details if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT NOT NULL,
                train_ratio INTEGER NOT NULL,
                test_ratio INTEGER NOT NULL,
                train_path TEXT NOT NULL,
                test_path TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                random_seed INTEGER NOT NULL
            );
        ''')
        self.conn.commit()
        cursor.close()

    def add_record(self, dataset_name, train_ratio, test_ratio, input_file, output_folder, random_seed):
        """Add a record for a dataset with a train/test split."""
        if train_ratio + test_ratio != 100:
            raise ValueError("The sum of train and test ratios must equal 100.")

        train_path, test_path = self.split_dataset(input_file, output_folder, train_ratio)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO dataset_records (dataset_name, train_ratio, test_ratio, train_path, test_path, timestamp, random_seed)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        ''', (dataset_name, train_ratio, test_ratio, train_path, test_path, timestamp, random_seed))
        self.conn.commit()
        cursor.close()

    def split_dataset(self, input_folder, output_folder, train_ratio, random_seed, pre_split_paths=None):
        """Split the image dataset into train and test parts, either by paths or by random seed."""
        if pre_split_paths:
            return pre_split_paths['train_path'], pre_split_paths['test_path']

        # Create output directories for train and test datasets
        train_path = os.path.join(output_folder, f"{uuid.uuid4()}_train")
        test_path = os.path.join(output_folder, f"{uuid.uuid4()}_test")
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        # Get all image files from the input directory
        all_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
        random.seed(random_seed)
        random.shuffle(all_files)
        
        # Calculate split index
        split_index = int(len(all_files) * (train_ratio / 100))
        
        # Move files to respective directories
        for file in all_files[:split_index]:
            shutil.move(file, os.path.join(train_path, os.path.basename(file)))
        for file in all_files[split_index:]:
            shutil.move(file, os.path.join(test_path, os.path.basename(file)))

        return train_path, test_path