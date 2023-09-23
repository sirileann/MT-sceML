from natsort import natsorted
import pandas as pd
import glob
import yaml

# list of config files to read
config_files = natsorted(glob.glob("./logs/model_v1/**/config.yaml"))

# empty list to store configuration data
config_data = []

# loop over config files and read data
for file in config_files:
    with open(file, 'r') as f:
        data = yaml.safe_load(f)
        config_data.append(data)

# convert configuration data to a pandas dataframe
df = pd.DataFrame(config_data)

# flatten the nested dictionaries using json_normalize
df = pd.json_normalize(df.to_dict(orient='records'))

# write dataframe to Excel file
writer = pd.ExcelWriter('config_data.xlsx', engine='openpyxl')
workbook = writer.book
df.to_excel(writer, sheet_name="Sheet1", index=False)
workbook.save('config_data.xlsx')
writer.close()
