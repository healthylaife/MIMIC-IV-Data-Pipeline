def create_Dict(self, meds, proc, out, chart, los):
    dataDic = {}
    print(los)
    labels_csv = pd.DataFrame(columns=["stay_id", "label"])
    labels_csv["stay_id"] = pd.Series(self.hids)
    labels_csv["label"] = 0
    #         print("# Unique gender",self.data.gender.nunique())
    #         print("# Unique ethnicity",self.data.ethnicity.nunique())
    #         print("# Unique insurance",self.data.insurance.nunique())

    for hid in self.hids:
        grp = self.data[self.data["stay_id"] == hid]
        dataDic[hid] = {
            "Cond": {},
            "Proc": {},
            "Med": {},
            "Out": {},
            "Chart": {},
            "ethnicity": grp["ethnicity"].iloc[0],
            "age": int(grp["Age"]),
            "gender": grp["gender"].iloc[0],
            "label": int(grp["label"]),
        }
        labels_csv.loc[labels_csv["stay_id"] == hid, "label"] = int(grp["label"])

        # print(static_csv.head())
    for hid in tqdm(self.hids):
        grp = self.data[self.data["stay_id"] == hid]
        demo_csv = grp[["Age", "gender", "ethnicity", "insurance"]]
        if not os.path.exists("./data/csv/" + str(hid)):
            os.makedirs("./data/csv/" + str(hid))
        demo_csv.to_csv("./data/csv/" + str(hid) + "/demo.csv", index=False)

        dyn_csv = pd.DataFrame()
        ###MEDS
        if self.feat_med:
            feat = meds["itemid"].unique()
            df2 = meds[meds["stay_id"] == hid]
            if df2.shape[0] == 0:
                amount = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
                amount = amount.fillna(0)
                amount.columns = pd.MultiIndex.from_product([["MEDS"], amount.columns])
            else:
                rate = df2.pivot_table(
                    index="start_time", columns="itemid", values="rate"
                )
                # print(rate)
                amount = df2.pivot_table(
                    index="start_time", columns="itemid", values="amount"
                )
                df2 = df2.pivot_table(
                    index="start_time", columns="itemid", values="stop_time"
                )
                # print(df2.shape)
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(
                    np.nan
                )
                df2 = pd.concat([df2, add_df])
                df2 = df2.sort_index()
                df2 = df2.ffill()
                df2 = df2.fillna(0)

                rate = pd.concat([rate, add_df])
                rate = rate.sort_index()
                rate = rate.ffill()
                rate = rate.fillna(-1)

                amount = pd.concat([amount, add_df])
                amount = amount.sort_index()
                amount = amount.ffill()
                amount = amount.fillna(-1)
                # print(df2.head())
                df2.iloc[:, 0:] = df2.iloc[:, 0:].sub(df2.index, 0)
                df2[df2 > 0] = 1
                df2[df2 < 0] = 0
                rate.iloc[:, 0:] = df2.iloc[:, 0:] * rate.iloc[:, 0:]
                amount.iloc[:, 0:] = df2.iloc[:, 0:] * amount.iloc[:, 0:]
                # print(df2.head())
                dataDic[hid]["Med"]["signal"] = df2.iloc[:, 0:].to_dict(orient="list")
                dataDic[hid]["Med"]["rate"] = rate.iloc[:, 0:].to_dict(orient="list")
                dataDic[hid]["Med"]["amount"] = amount.iloc[:, 0:].to_dict(
                    orient="list"
                )

                feat_df = pd.DataFrame(columns=list(set(feat) - set(amount.columns)))
                #                 print(feat)
                #                 print(amount.columns)
                #                 print(amount.head())
                amount = pd.concat([amount, feat_df], axis=1)

                amount = amount[feat]
                amount = amount.fillna(0)
                #                 print(amount.columns)
                amount.columns = pd.MultiIndex.from_product([["MEDS"], amount.columns])

            if dyn_csv.empty:
                dyn_csv = amount
            else:
                dyn_csv = pd.concat([dyn_csv, amount], axis=1)

        ###PROCS
        if self.feat_proc:
            feat = proc["itemid"].unique()
            df2 = proc[proc["stay_id"] == hid]
            if df2.shape[0] == 0:
                df2 = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
                df2 = df2.fillna(0)
                df2.columns = pd.MultiIndex.from_product([["PROC"], df2.columns])
            else:
                df2["val"] = 1
                # print(df2)
                df2 = df2.pivot_table(
                    index="start_time", columns="itemid", values="val"
                )
                # print(df2.shape)
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(
                    np.nan
                )
                df2 = pd.concat([df2, add_df])
                df2 = df2.sort_index()
                df2 = df2.fillna(0)
                df2[df2 > 0] = 1
                # print(df2.head())
                dataDic[hid]["Proc"] = df2.to_dict(orient="list")

                feat_df = pd.DataFrame(columns=list(set(feat) - set(df2.columns)))
                df2 = pd.concat([df2, feat_df], axis=1)

                df2 = df2[feat]
                df2 = df2.fillna(0)
                df2.columns = pd.MultiIndex.from_product([["PROC"], df2.columns])

            if dyn_csv.empty:
                dyn_csv = df2
            else:
                dyn_csv = pd.concat([dyn_csv, df2], axis=1)

        ###OUT
        if self.feat_out:
            feat = out["itemid"].unique()
            df2 = out[out["stay_id"] == hid]
            if df2.shape[0] == 0:
                df2 = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
                df2 = df2.fillna(0)
                df2.columns = pd.MultiIndex.from_product([["OUT"], df2.columns])
            else:
                df2["val"] = 1
                df2 = df2.pivot_table(
                    index="start_time", columns="itemid", values="val"
                )
                # print(df2.shape)
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(
                    np.nan
                )
                df2 = pd.concat([df2, add_df])
                df2 = df2.sort_index()
                df2 = df2.fillna(0)
                df2[df2 > 0] = 1
                # print(df2.head())
                dataDic[hid]["Out"] = df2.to_dict(orient="list")

                feat_df = pd.DataFrame(columns=list(set(feat) - set(df2.columns)))
                df2 = pd.concat([df2, feat_df], axis=1)

                df2 = df2[feat]
                df2 = df2.fillna(0)
                df2.columns = pd.MultiIndex.from_product([["OUT"], df2.columns])

            if dyn_csv.empty:
                dyn_csv = df2
            else:
                dyn_csv = pd.concat([dyn_csv, df2], axis=1)

        ###CHART
        if self.feat_chart:
            feat = chart["itemid"].unique()
            df2 = chart[chart["stay_id"] == hid]
            if df2.shape[0] == 0:
                val = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
                val = val.fillna(0)
                val.columns = pd.MultiIndex.from_product([["CHART"], val.columns])
            else:
                val = df2.pivot_table(
                    index="start_time", columns="itemid", values="valuenum"
                )
                df2["val"] = 1
                df2 = df2.pivot_table(
                    index="start_time", columns="itemid", values="val"
                )
                # print(df2.shape)
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(
                    np.nan
                )
                df2 = pd.concat([df2, add_df])
                df2 = df2.sort_index()
                df2 = df2.fillna(0)

                val = pd.concat([val, add_df])
                val = val.sort_index()
                if self.impute == "Mean":
                    val = val.ffill()
                    val = val.bfill()
                    val = val.fillna(val.mean())
                elif self.impute == "Median":
                    val = val.ffill()
                    val = val.bfill()
                    val = val.fillna(val.median())
                val = val.fillna(0)

                df2[df2 > 0] = 1
                df2[df2 < 0] = 0
                # print(df2.head())
                dataDic[hid]["Chart"]["signal"] = df2.iloc[:, 0:].to_dict(orient="list")
                dataDic[hid]["Chart"]["val"] = val.iloc[:, 0:].to_dict(orient="list")

                feat_df = pd.DataFrame(columns=list(set(feat) - set(val.columns)))
                val = pd.concat([val, feat_df], axis=1)

                val = val[feat]
                val = val.fillna(0)
                val.columns = pd.MultiIndex.from_product([["CHART"], val.columns])

            if dyn_csv.empty:
                dyn_csv = val
            else:
                dyn_csv = pd.concat([dyn_csv, val], axis=1)

        # Save temporal data to csv
        dyn_csv.to_csv("./data/csv/" + str(hid) + "/dynamic.csv", index=False)

        ##########COND#########
        if self.feat_cond:
            feat = self.cond["new_icd_code"].unique()
            grp = self.cond[self.cond["stay_id"] == hid]
            if grp.shape[0] == 0:
                dataDic[hid]["Cond"] = {"fids": list(["<PAD>"])}
                feat_df = pd.DataFrame(np.zeros([1, len(feat)]), columns=feat)
                grp = feat_df.fillna(0)
                grp.columns = pd.MultiIndex.from_product([["COND"], grp.columns])
            else:
                dataDic[hid]["Cond"] = {"fids": list(grp["new_icd_code"])}
                grp["val"] = 1
                grp = grp.drop_duplicates()
                grp = grp.pivot(
                    index="stay_id", columns="new_icd_code", values="val"
                ).reset_index(drop=True)
                feat_df = pd.DataFrame(columns=list(set(feat) - set(grp.columns)))
                grp = pd.concat([grp, feat_df], axis=1)
                grp = grp.fillna(0)
                grp = grp[feat]
                grp.columns = pd.MultiIndex.from_product([["COND"], grp.columns])
        grp.to_csv("./data/csv/" + str(hid) + "/static.csv", index=False)
        labels_csv.to_csv("./data/csv/labels.csv", index=False)

    ######SAVE DICTIONARIES##############
    metaDic = {"Cond": {}, "Proc": {}, "Med": {}, "Out": {}, "Chart": {}, "LOS": {}}
    metaDic["LOS"] = los
    with open("./data/dict/dataDic", "wb") as fp:
        pickle.dump(dataDic, fp)

    with open("./data/dict/hadmDic", "wb") as fp:
        pickle.dump(self.hids, fp)

    with open("./data/dict/ethVocab", "wb") as fp:
        pickle.dump(list(self.data["ethnicity"].unique()), fp)
        self.eth_vocab = self.data["ethnicity"].nunique()

    with open("./data/dict/ageVocab", "wb") as fp:
        pickle.dump(list(self.data["Age"].unique()), fp)
        self.age_vocab = self.data["Age"].nunique()

    with open("./data/dict/insVocab", "wb") as fp:
        pickle.dump(list(self.data["insurance"].unique()), fp)
        self.ins_vocab = self.data["insurance"].nunique()

    if self.feat_med:
        with open("./data/dict/medVocab", "wb") as fp:
            pickle.dump(list(meds["itemid"].unique()), fp)
        self.med_vocab = meds["itemid"].nunique()
        metaDic["Med"] = self.med_per_adm

    if self.feat_out:
        with open("./data/dict/outVocab", "wb") as fp:
            pickle.dump(list(out["itemid"].unique()), fp)
        self.out_vocab = out["itemid"].nunique()
        metaDic["Out"] = self.out_per_adm

    if self.feat_chart:
        with open("./data/dict/chartVocab", "wb") as fp:
            pickle.dump(list(chart["itemid"].unique()), fp)
        self.chart_vocab = chart["itemid"].nunique()
        metaDic["Chart"] = self.chart_per_adm

    if self.feat_cond:
        with open("./data/dict/condVocab", "wb") as fp:
            pickle.dump(list(self.cond["new_icd_code"].unique()), fp)
        self.cond_vocab = self.cond["new_icd_code"].nunique()
        metaDic["Cond"] = self.cond_per_adm

    if self.feat_proc:
        with open("./data/dict/procVocab", "wb") as fp:
            pickle.dump(list(proc["itemid"].unique()), fp)
        self.proc_vocab = proc["itemid"].nunique()
        metaDic["Proc"] = self.proc_per_adm

    with open("./data/dict/metaDic", "wb") as fp:
        pickle.dump(metaDic, fp)
