#%%
import pandas as pd
import regex as re
#%%
df_alarm = pd.read_csv('/Users/hamzagorgulu/Desktop/thesis/Waris_COMP/research/tupras_analysis/data/processed/alarms/final/final-all-months-alarms-with-day.csv')
#%%
df_alarm.head()
# %% I will filter SourceName
pattern = "47...2...|47...3...|47..2...|47..3...|^18..|^18...|^19..|^19...|CEMS|F201|F21|^TEST.+...2|^TEST.+...3|47.+48|^01|18"
filter_list = []
for source in df_alarm['SourceName'].unique():
    if re.match(pattern, source):
        filter_list.append(source)
print(len(filter_list))
print(len(df_alarm['SourceName'].unique()))

# %% delete the complete row with the filtered SourceName
df_alarm_filtered = df_alarm.drop([df_alarm.index[i] for i in range(len(df_alarm)) if df_alarm['SourceName'][i] in filter_list])
df_alarm_filtered.shape
# %%
filtered_sourcename = df_alarm_filtered.SourceName.unique()
df_alarm.shape
# %% save filtered df
df_alarm_filtered.to_csv('/Users/hamzagorgulu/Desktop/thesis/Waris_COMP/research/tupras_analysis/data/processed/alarms/final/final-all-months-alarms-with-day-filtered.csv', index=False)

# %% op actions
df_op = pd.read_csv("/Users/hamzagorgulu/Desktop/thesis/Waris_COMP/research/tupras_analysis/data/processed/operator-actions/final/final-all-month-actions.csv")
df_op.head()
# %%
df_op.SourceName.nunique()
# %%
pattern = "47...2...|47...3...|47..2...|47..3...|^18..|^18...|^19..|^19...|CEMS|F201|F21|^TEST.+...2|^TEST.+...3|47.+48|^01|18"
filter_list_op = []
for source in df_op['SourceName'].unique():
    if re.match(pattern, source):
        filter_list_op.append(source)
# %%
print(filter_list_op)
# %% filter op
print(df_op.shape)
df_op_filtered = df_op.drop([df_op.index[i] for i in range(len(df_op)) if df_op['SourceName'][i] in filter_list_op])
print(df_op_filtered.shape)

# %% export to csv
df_op_filtered.to_csv('/Users/hamzagorgulu/Desktop/thesis/Waris_COMP/research/tupras_analysis/data/processed/operator-actions/final/final-all-month-actions-filtered.csv', index=False)

# %% take only unique SourceName
unique_op = df_op_filtered.SourceName.unique()
# save as csv
pd.DataFrame(unique_op).to_csv('/Users/hamzagorgulu/Desktop/thesis/Waris_COMP/research/tupras_analysis/data/processed/operator-actions/final/unique_actions_sourcename.csv', index=False)
# %%
unique_alarm = df_alarm_filtered.SourceName.unique()
pd.DataFrame(unique_alarm).to_csv('/Users/hamzagorgulu/Desktop/thesis/Waris_COMP/research/tupras_analysis/data/processed/operator-actions/final/unique_alarms_sourcename.csv', index=False)

# %% merge two list using append in a function
def merge_list(list1, list2):
    return list1 + list2
# merge two listNode and return a list in function
def merge_listNode(list1, list2):
    return list1 + list2
