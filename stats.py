import os
import json

stats_number = {
    'original_train': -1,
    'original_test': -1,
    'total_number_of_labels': -1,
    'preprocessed_train': -1,
    'preprocessed_test': -1,
    # the number of missing preprocessed train images
    'missing_preprocessed_train': -1,
    # the number of missing preprocessed test images
    'missing_preprocessed_test': -1,
    # the names of the missing preprocessed test images
    'missing_preprocessed_test_images': [],
    # the number of preprocessed train json files
    'preprocessed_train_json': -1,
    # the number of preprocessed test json files
    'preprocessed_test_json': -1,
    # the number of missing preprocessed train json files
    'missing_preprocessed_train_json': -1,
    # the number of missing preprocessed test instances
    'missing_preprocessed_test_json': -1,
    # the names of the missing preprocessed train json files
    'missing_preprocessed_train_json_files': [],
}


def write_dict_as_md_table(f, title, data_dict):
    f.write(f'## {title}\n\n')
    keys = list(data_dict.keys())
    values = list(data_dict.values())
    num_items = len(keys)
    num_columns = 3
    num_rows = (num_items + num_columns - 1) // num_columns

    f.write('| Label 1 | Count 1 | | Label 2 | Count 2 | | Label 3 | Count 3 |\n')
    f.write('|---------|---------|-|---------|---------|-|---------|---------|\n')

    for row in range(num_rows):
        for col in range(num_columns):
            index = row + col * num_rows
            if index < num_items:
                f.write(f'| {keys[index]} | {values[index]} | ')
            else:
                f.write('|   |   | ')
        f.write('\n')
    f.write('\n')


class StatsGenerator:
    def __init__(
            self,
            original_train_path,
            processed_train_images_path,
            processed_coordinates_train_path,
            original_test_path,
            processed_test_images_path,
            processed_coordinates_test_path,
            output_file):
        self.original_train_path = original_train_path
        self.original_test_path = original_test_path

        self.processed_train_images_path = processed_train_images_path
        self.processed_test_images_path = processed_test_images_path

        self.processed_coordinates_train_path = processed_coordinates_train_path
        self.processed_coordinates_test_path = processed_coordinates_test_path

        self.output_file = output_file

    def generate_stats(self):
        global stats_number
        with open(self.output_file, 'w') as f:
            f.write('# Original data statistics\n')
            # Train data
            original_train_data = self._get_train_images_data_stats(
                self.original_train_path)
            write_dict_as_md_table(
                f, 'Original train data', original_train_data)

            stats_number['original_train'] = sum(original_train_data.values())
            stats_number['total_number_of_labels'] = len(
                original_train_data.keys())
            f.write('### Total number of labels: {}\n'.format(
                stats_number['total_number_of_labels']))
            f.write(
                f'### Total number of train images: {
                    stats_number["original_train"]}\n\n')

            # Test data
            original_test_data = self._get_test_images_data_stats(
                self.original_test_path)
            write_dict_as_md_table(f, 'Original test data', original_test_data)

            stats_number['original_test'] = sum(original_test_data.values())
            f.write(
                f'### Total number of test images: {
                    stats_number["original_test"]}\n\n')

            # Train json data
            f.write('# Preprocessed coordinates data statistics\n')
            preprocessed_train_json_data = self._get_json_train_data_stats(
                self.processed_coordinates_train_path)
            write_dict_as_md_table(
                f, 'Preprocessed train json data', preprocessed_train_json_data)
            stats_number['preprocessed_train_json'] = sum(
                preprocessed_train_json_data.values())
            stats_number['missing_preprocessed_train_json'] = stats_number['original_train'] - \
                stats_number['preprocessed_train_json']
            f.write('### Total number of labels: {}\n'.format(
                stats_number['total_number_of_labels']))
            f.write(
                f'### Total number of preprocessed train json files: {
                    stats_number["preprocessed_train_json"]}\n')
            f.write(
                f'### Number of missing preprocessed train json files: {
                    stats_number["missing_preprocessed_train_json"]}\n\n')

            # Test json data
            preprocessed_test_json_data = self._get_json_test_data_stats(
                self.processed_coordinates_test_path)
            write_dict_as_md_table(
                f, 'Preprocessed test json data', preprocessed_test_json_data)
            stats_number['preprocessed_test_json'] = sum(
                preprocessed_test_json_data.values())
            f.write(
                f'### Total number of test json files: {
                    stats_number["preprocessed_test_json"]}\n\n')

            f.write(f'### The missing preproccessed test instances: ')
            missing_test_labels = set(
                original_test_data.keys()) - set(preprocessed_test_json_data.keys())
            stats_number['missing_preprocessed_test_instances'] = missing_test_labels
            stats_number['missing_preprocessed_test_json'] = len(
                missing_test_labels)
            f.write(
                f'{missing_test_labels} : {
                    stats_number['missing_preprocessed_test_json']} missing instances\n\n')

            # Compare the original and preprocessed data
            f.write(
                '\n# Comparison between original and preprocessed coordinates data\n')
            sorted_original_train_data = dict(
                sorted(original_train_data.items()))
            for label in sorted_original_train_data:
                if label in preprocessed_train_json_data:
                    f.write(f'### Label: {label}\n')
                    f.write(
                        f'- Original train data: {original_train_data[label]}\n')
                    f.write(
                        f'- Preprocessed train json data: {preprocessed_train_json_data[label]}\n')
                    f.write(
                        f'- Number of missing preprocessed train json data for this label: {
                            original_train_data[label] -
                            preprocessed_train_json_data[label]}\n')
                    f.write(
                        f'- Percentage of preprocessed train json data: {
                            preprocessed_train_json_data[label] /
                            original_train_data[label] *
                            100:.2f}%\n\n')

            f.write(
                f'## The average percentage of preprocessed train json data: {
                    sum(
                        preprocessed_train_json_data.values()) / sum(
                        original_train_data.values()) * 100:.2f}%\n\n')
            f.write(
                f'## The average percentage of preprocessed test json data: {
                    sum(
                        preprocessed_test_json_data.values()) / sum(
                        original_test_data.values()) * 100:.2f}%\n\n')

            f.write('# Preprocessed images data statistics\n')
            # Train data
            preprocessed_train_data = self._get_train_images_data_stats(
                self.processed_train_images_path)
            write_dict_as_md_table(
                f, 'Preprocessed train data', preprocessed_train_data)

            stats_number['preprocessed_train'] = sum(
                preprocessed_train_data.values())
            stats_number['missing_preprocessed_train'] = stats_number['original_train'] - \
                stats_number['preprocessed_train']
            f.write('### Total number of labels: {}\n'.format(
                stats_number['total_number_of_labels']))
            f.write(
                f'### Total number of preprocessed train images: {
                    stats_number["preprocessed_train"]}\n')
            f.write(
                f'### Number of missing preprocessed train images: {
                    stats_number["missing_preprocessed_train"]}\n\n')

            # Test data
            preprocessed_test_data = self._get_test_images_data_stats(
                self.processed_test_images_path)
            write_dict_as_md_table(
                f, 'Preprocessed test data', preprocessed_test_data)

            stats_number['preprocessed_test'] = sum(
                preprocessed_test_data.values())
            f.write(
                f'### Total number of test images: {
                    stats_number["preprocessed_test"]}\n\n')

            f.write(f'### The missing preproccessed test images: ')
            missing_test_labels = set(
                original_test_data.keys()) - set(preprocessed_test_data.keys())
            stats_number['missing_preprocessed_test_images'] = missing_test_labels
            stats_number['missing_preprocessed_test'] = len(
                missing_test_labels)
            f.write(
                f'{missing_test_labels} : {
                    stats_number['missing_preprocessed_test']} missing images\n\n')

            # Compare the original and preprocessed data
            f.write('\n# Comparison between original and preprocessed images data\n')
            sorted_original_train_data = dict(
                sorted(original_train_data.items()))
            for label in sorted_original_train_data:
                if label in preprocessed_train_data:
                    f.write(f'### Label: {label}\n')
                    f.write(
                        f'- Original train data: {original_train_data[label]}\n')
                    f.write(
                        f'- Preprocessed train data: {preprocessed_train_data[label]}\n')
                    f.write(
                        f'- Number of missing preprocessed train data for this label: {
                            original_train_data[label] -
                            preprocessed_train_data[label]}\n')
                    f.write(
                        f'- Percentage of preprocessed train data: {
                            preprocessed_train_data[label] /
                            original_train_data[label] *
                            100:.2f}%\n\n')

            f.write(
                f'## The average procentage of preprocessed train data: {
                    sum(
                        preprocessed_train_data.values()) / sum(
                        original_train_data.values()) * 100:.2f}%\n\n')
            f.write(
                f'## The average procentage of preprocessed test data: {
                    sum(
                        preprocessed_test_data.values()) / sum(
                        original_test_data.values()) * 100:.2f}%\n\n')

    def _get_train_images_data_stats(self, path):
        data_stats = {}
        for label in os.listdir(path):
            for image in os.listdir(os.path.join(path, label)):
                if label not in data_stats:
                    data_stats[label] = 1
                else:
                    data_stats[label] += 1
        return data_stats

    def _get_test_images_data_stats(self, path):
        data_stats = {}
        for image in os.listdir(path):
            label = image.split('.')[0]
            if label not in data_stats:
                data_stats[label] = 1
            else:
                data_stats[label] += 1
        return data_stats

    def _get_json_train_data_stats(self, path):
        data_stats = {}
        for json_file in os.listdir(path):
            label = json_file.split('.')[0]
            with open(os.path.join(path, json_file), 'r') as file:
                data = json.load(file)
                data_stats[label] = len(data)
        return data_stats

    def _get_json_test_data_stats(self, path):
        data_stats = {}
        data = json.load(open(path))
        for item in data:
            label = item['label']
            if label not in data_stats:
                data_stats[label] = 1
            else:
                data_stats[label] += 1
        return data_stats


if __name__ == "__main__":
    original_train_path = 'data/asl_alphabet_train'
    original_test_path = 'data/asl_alphabet_test'

    processed_images_train_path = 'processed_images/train'
    processed_images_test_path = 'processed_images/test'

    processed_coordinates_train_path = 'processed_coordinates/train'
    processed_coordinates_test_path = 'processed_coordinates/test/test.json'

    output_file = 'stats.md'

    stats_generator = StatsGenerator(
        original_train_path,
        processed_images_train_path,
        processed_coordinates_train_path,
        original_test_path,
        processed_images_test_path,
        processed_coordinates_test_path,
        output_file)
    stats_generator.generate_stats()
