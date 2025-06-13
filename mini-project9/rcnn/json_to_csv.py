import json
import os
import argparse


def convert_labelme_folder_to_csv(input_folder):
    output_folder = f"{input_folder}_csv"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(input_folder, filename)
            csv_filename = os.path.splitext(filename)[0] + ".csv"
            csv_path = os.path.join(output_folder, csv_filename)

            with open(json_path, 'r') as file:
                data = json.load(file)

            shapes = data.get('shapes', [])
            with open(csv_path, 'w') as csv_file:
                csv_file.write("class,xmin,ymin,xmax,ymax\n")
                for shape in shapes:
                    label = shape.get('label', 'unknown')
                    points = shape.get('points', [])

                    if not points:
                        continue

                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    xmin = int(min(x_coords))
                    ymin = int(min(y_coords))
                    xmax = int(max(x_coords))
                    ymax = int(max(y_coords))

                    csv_file.write(f"{label},{xmin},{ymin},{xmax},{ymax}\n")

    print(f"✅ Converted CSVs saved to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description='Convert LabelMe JSON to CSV.')
    parser.add_argument('-d', '--directory', type=str, required=True, help='Directory containing LabelMe JSON files.')
    args = parser.parse_args()
    convert_labelme_folder_to_csv(args.directory)


if __name__ == "__main__":
    main()
