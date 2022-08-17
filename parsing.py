import pandas as pd

custom_csv_file_df = pd.read_csv('/home/ubuntu/Sketch-Recommendation/data/custom_class_list.csv')
object_detection_csv = pd.read_csv('/home/ubuntu/Sketch-Recommendation/data/2017_object_detection_synset_ID.csv')

custom_synset = custom_csv_file_df[['synset_ID', 'class_name']]
object_detection_synset = object_detection_csv[['synset_ID', 'class_name']]

# print(custom_synset)

set_custom_synset = set(custom_synset['synset_ID'])
set_custom_name_synset = set(custom_synset['class_name'])


set_object_detection = set(object_detection_synset['synset_ID'])
set_object_detection_name = set(object_detection_synset['class_name'])

subtract = set_custom_synset.difference(set_object_detection)
print(subtract)
print(len(subtract))

subtract_name = set_custom_name_synset.difference(set_object_detection_name)
print(subtract_name)
print(len(subtract_name))

# sub_list = []
# for item in subtract:
#     sub_list.append(custom_synset[item])
# print(sub_list)