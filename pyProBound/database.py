import json
import pandas as pd
import os
import numpy as np

current = os.path.split(os.path.realpath(__file__))[0]
jardir = f"{current}"
motifcentral_json = f"{jardir}/MotifCentral.v1.0.0.json"

tax_id_map = {
    10090: "Mus musculus",
    9606: "Homo sapiens",
    7227: "Drosophila melanogaster",
    94885: "Homo sapiens"
}


class MotifCentral:
    def __init__(self):
        # loads database
        data = []
        for item in json.load(open(motifcentral_json)):
            fit_id = item["metadata"]["fit_id"]

            gene_symbols = [x["gene_symbol"] for x in item["metadata"]["factors"]]
            tax_ids = [x["tax_id"] for x in item["metadata"]["factors"]][0]
            gene_names = ",".join([x["gene_name"] for x in item["metadata"]["factors"] if x["gene_name"] is not None])

            publications = [x[0] for x in item["metadata"]["experiments"]]
            studies = [f"{x[0]}-{x[1]}" for x in item["metadata"]["experiments"]]

            data.append([fit_id, gene_symbols, tax_ids, gene_names, publications, studies])

        self.data = pd.DataFrame(data,
                                 columns=["model_id",
                                          "gene_symbols",
                                          "tax_id",
                                          "gene_names",
                                          "publications",
                                          "studies"])
        self.data["taxa"] = self.data["tax_id"].map(tax_id_map)

    def get_options(self):
        """
        Returns unique values in the data for queryable features.
        :return: a dictionary with possible values
        """
        options = {}
        for col in ["taxa", "publications", "gene_symbols"]:
            examples = self.data[col].explode().unique()
            options[col] = examples

        return options

    def get_model_id_desc(self, model_id):
        """
        Get the row with model description given model ID.
        :param model_id: model ID
        :return:
        """
        if model_id not in self.data["model_id"].values:
            raise Exception(f"Model ID {model_id} is not in the database.")
        return self.data[self.data["model_id"] == model_id]

    def filter(self,
               taxa=None,
               publications=None,
               gene_symbols=None,
               ):
        """
        Get a specified subset of the database. Has "contains" logic.
        :param taxa: list of taxa to return. If None, return all
        :param publications: list of publications to return. If None, return all possible
        :param gene_symbols: list of gene symbols to return. If None, return all possible
        :return: dataframe with results.
        """
        mask = np.ones(len(self.data), dtype=bool)

        # single value column
        if taxa is not None:
            taxamask = np.zeros(len(self.data), dtype=bool)
            for t in taxa:
                taxamask = taxamask | (self.data["taxa"] == t)
            mask = taxamask & mask

        # multi value column
        for col, query in zip(["publications", "gene_symbols"],
                              [publications, gene_symbols]):
            if query is None:
                continue
            colmask = np.zeros(len(self.data), dtype=bool)
            for q in query:
                qmask = self.data[col].map({q}.issubset)
                colmask = colmask | qmask
            mask = colmask & mask

        view = self.data[mask].copy()
        return view
