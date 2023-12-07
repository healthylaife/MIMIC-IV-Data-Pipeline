# def mortality_length(data, include_time, predW):
#     los = include_time
#     data = data[(data["los"] >= include_time + predW)]
#     hids = data["hadm_id"].unique()

#     if self.feat_cond:
#         self.cond = self.cond[self.cond["hadm_id"].isin(hids)]

#     data["los"] = include_time
#     ###MEDS
#     if self.feat_med:
#         self.meds = self.meds[self.meds["hadm_id"].isin(self.data["hadm_id"])]
#         self.meds = self.meds[self.meds["start_time"] <= include_time]
#         self.meds.loc[
#             self.meds.stop_time > include_time, "stop_time"
#         ] = include_time

#     ###PROCS
#     if self.feat_proc:
#         self.proc = self.proc[self.proc["hadm_id"].isin(self.data["hadm_id"])]
#         self.proc = self.proc[self.proc["start_time"] <= include_time]

#     ###LAB
#     if self.feat_lab:
#         self.labs = self.labs[self.labs["hadm_id"].isin(self.data["hadm_id"])]
#         self.labs = self.labs[self.labs["start_time"] <= include_time]

#     self.los = include_time

# def los_length(self, include_time):
#     self.los = include_time
#     self.data = self.data[(self.data["los"] >= include_time)]
#     self.hids = self.data["hadm_id"].unique()

#     if self.feat_cond:
#         self.cond = self.cond[self.cond["hadm_id"].isin(self.data["hadm_id"])]

#     self.data["los"] = include_time
#     ###MEDS
#     if self.feat_med:
#         self.meds = self.meds[self.meds["hadm_id"].isin(self.data["hadm_id"])]
#         self.meds = self.meds[self.meds["start_time"] <= include_time]
#         self.meds.loc[
#             self.meds.stop_time > include_time, "stop_time"
#         ] = include_time

#     ###PROCS
#     if self.feat_proc:
#         self.proc = self.proc[self.proc["hadm_id"].isin(self.data["hadm_id"])]
#         self.proc = self.proc[self.proc["start_time"] <= include_time]

#     ###LAB
#     if self.feat_lab:
#         self.labs = self.labs[self.labs["hadm_id"].isin(self.data["hadm_id"])]
#         self.labs = self.labs[self.labs["start_time"] <= include_time]

#     # self.los=include_time

# def readmission_length(self, include_time):
#     self.los = include_time
#     self.data = self.data[(self.data["los"] >= include_time)]
#     self.hids = self.data["hadm_id"].unique()
#     if self.feat_cond:
#         self.cond = self.cond[self.cond["hadm_id"].isin(self.data["hadm_id"])]
#     self.data["select_time"] = self.data["los"] - include_time
#     self.data["los"] = include_time

#     ####Make equal length input time series and remove data for pred window if needed

#     ###MEDS
#     if self.feat_med:
#         self.meds = self.meds[self.meds["hadm_id"].isin(self.data["hadm_id"])]
#         self.meds = pd.merge(
#             self.meds,
#             self.data[["hadm_id", "select_time"]],
#             on="hadm_id",
#             how="left",
#         )
#         self.meds["stop_time"] = self.meds["stop_time"] - self.meds["select_time"]
#         self.meds["start_time"] = self.meds["start_time"] - self.meds["select_time"]
#         self.meds = self.meds[self.meds["stop_time"] >= 0]
#         self.meds.loc[self.meds.start_time < 0, "start_time"] = 0

#     ###PROCS
#     if self.feat_proc:
#         self.proc = self.proc[self.proc["hadm_id"].isin(self.data["hadm_id"])]
#         self.proc = pd.merge(
#             self.proc,
#             self.data[["hadm_id", "select_time"]],
#             on="hadm_id",
#             how="left",
#         )
#         self.proc["start_time"] = self.proc["start_time"] - self.proc["select_time"]
#         self.proc = self.proc[self.proc["start_time"] >= 0]

#     ###LABS
#     if self.feat_lab:
#         self.labs = self.labs[self.labs["hadm_id"].isin(self.data["hadm_id"])]
#         self.labs = pd.merge(
#             self.labs,
#             self.data[["hadm_id", "select_time"]],
#             on="hadm_id",
#             how="left",
#         )
#         self.labs["start_time"] = self.labs["start_time"] - self.labs["select_time"]
#         self.labs = self.labs[self.labs["start_time"] >= 0]
