import jpype
import jpype.imports
import atexit
import numpy as np
import pandas as pd
import os

current = os.path.split(os.path.realpath(__file__))[0]
jardir = f"{current}/jardir"
print(jardir)
generalSchemaFile = f"{jardir}/schema.general.json"
jpype.startJVM(classpath=[f'{jardir}/ProBound-jar-with-dependencies.jar'])


def __closeJVM():
    jpype.shutdownJVM()


atexit.register(__closeJVM)


from proBoundTools import Toolbox


class ProBoundModel:
    """
    Class representing a ProBound model.
    """

    def __create_java_arraylist(self, iterable, throw_for_none=True):
        javalist = jpype.java.util.ArrayList()
        if (iterable is None) and throw_for_none:
            raise Exception("Trying to iterate over None.")
        elif (iterable is None):
            # create empty ArrayList
            return javalist
        for item in iterable:
            javalist.add(item)

        return javalist

    def __create_numpy(self, javaarray):
        # javaarray -- possibly array of arrays
        return np.vstack([np.array(x) for x in javaarray])

    __profile_aggr = {
        "sum": lambda desc: np.sum(desc, axis=1),
        "mean": lambda desc: np.mean(desc, axis=1),
        "max": lambda desc: np.max(desc, axis=1),
        "forward": lambda desc: desc[:, 0],
    }

    # backend -- toolbox object
    def __init__(self, source,
                 motifcentral=False,
                 fitjson=False,
                 iLine=-1,
                 withN=True,
                 bindingMode=None):
        """
        :param source: path to a json file (fitjson=True) or model number in MotifCentral (motifcentral=True)
        :boolean motifcentral: bool, load source from motifcentral
        :boolean fitjson: bool, load source from local fit json model
        :int iLine: for fitjson -- use the model on iLine. Default: -1 (line with the smallest -log(likelihood))
        :boolean withN: add N to scoring alphabet
        :int bindingMode: select binding mode. Default: use all binding modes
        """
        self.t = Toolbox(generalSchemaFile, False)
        if motifcentral:
            self.t.loadMotifCentralModel(source)
        elif fitjson:
            self.t.loadFitLine(source, generalSchemaFile, iLine)
            self.t.buildConsensusModel()
        else:
            self.t.loadMotifCentralModel(source)

        if withN:
            self.t.addNScoring()
        if bindingMode is not None:
            self.t.selectBindingMode(bindingMode)

        self.current_sequences = None

    def select_binding_mode(self, bindingMode):
        """
        Selects binding mode to use in model. By default, all are used.
        :param bindingMode: integer identifier for the binding mode
        """
        self.t.selectBindingMode(bindingMode)

    def __group_sequences(self, sequences):
        # group sequences by size, yield groups
        current_sequences = pd.DataFrame(sequences, columns=["seq"])
        current_sequences["len"] = current_sequences["seq"].str.len()
        for len_value, group in current_sequences.groupby("len"):
            yield len_value, group["seq"]

    def __reorder_results(self, results, indexes):
        # sort results to the default order of sequences
        # is a list -- possibility of a ragged sequence
        # results can have variable dimensions
        a = np.argsort(indexes)
        output = [results[ai] for ai in a]
        return output

    def score_affinity_sum(self, sequences, modifications=None):
        """
        Calculate affinity sum for given sequences.
        :param sequences: iterable of sequences (list, numpy array). Must be uppercase.
        :param modifications: not implemented
        :return: affinity sum value for each sequence in input (numpy array)
        """
        # sequences -- iterable of strings, can be different sizes
        # modifications
        # output: a numpy array (no of sequences) X (no of experiment rounds)
        indexes, results = [], []
        for len_value, seq_size_group in self.__group_sequences(sequences):
            r = self.__score_affinity_sum_same_size(seq_size_group, modifications)
            indexes.extend(seq_size_group.index)
            results.extend(r)
        results = np.array(results)
        return np.vstack(self.__reorder_results(results, np.array(indexes)))

    def score_binding_mode_scores(self, sequences,
                                  modifications=None,
                                  score_format="sum", profile_aggregate=None, ):
        """
        Calculate scores for selected binding mode.
        :param sequences: iterable of sequences (list, numpy array). Must be uppercase.
        :param modifications: not implemented
        :param score_format: Format of the score -- sum/mean/max/profile
        :param profile_aggregate: if score_format == profile, option to aggregate forward/reverse (for double-strand).
                    Options: sum/mean/max/forward/None. Default: None (no aggregation done).
        :return: iterable of scores.
        """
        # returns a numpy array with results
        # if score format is an aggregate function (sum/mean/max) -- len(sequences) X model_binding_modes
        # if score format is profile --  list of items for each sequence:  model_binding_modes X slides X 2(forward, reverse)

        indexes, results = [], []
        for len_value, seq_size_group in self.__group_sequences(sequences):
            r = self.__score_binding_mode_scores_same_size(seq_size_group,
                                                           modifications=modifications,
                                                           score_format=score_format,
                                                           profile_aggregate=profile_aggregate
                                                           )
            indexes.extend(seq_size_group.index)
            results.extend(r)
        output = self.__reorder_results(results, np.array(indexes))
        if score_format != "profile":  # unsupported are already taken care of
            return np.array(output)
        return output  # profile -- returns list, can be ragged

    def __score_affinity_sum_same_size(self, sequences, modifications=None):
        # sequences -- iterable of strings, must be same size
        # modifications
        # output: a numpy array (no of sequences) X (no of experiment rounds)
        sequences_java = self.__create_java_arraylist(sequences)
        modifications_java = self.__create_java_arraylist(modifications,
                                                          throw_for_none=False)
        # create
        self.t.inputExistingSequnces(sequences_java, modifications_java)
        count_table = self.t.getCountTable()
        result = count_table.calculateAlphaTable()
        return self.__create_numpy(result)

    def __score_binding_mode_scores_same_size(self,
                                              sequences,
                                              modifications=None,
                                              score_format="sum",
                                              profile_aggregate=None,
                                              ):
        # sequences -- iterable of strings, must be same size
        # returns a numpy array with results
        # if score format is an aggregate function (sum/mean/max) -- len(sequences) X model_binding_modes
        # if score format is profile and profile_aggregate is None --
        #                   -- len(sequences) X model_binding_modes X slides X 2 (forward, reverse)
        # if score format is profile and profile_aggregate is sum/max/mean --
        #                   -- len(sequences) X model_binding_modes X slides
        sequences_java = self.__create_java_arraylist(sequences)
        modifications_java = self.__create_java_arraylist(modifications,
                                                          throw_for_none=False)
        self.t.inputExistingSequnces(sequences_java, modifications_java)
        count_table = self.t.getCountTable()
        if score_format in ["sum", "mean", "max"]:
            # get array of values for all binding modes in the model
            bm_results = count_table.calculateAggregateBindingModeAlphas(score_format)
            result = np.hstack([self.__create_numpy(bm_array) for bm_array in bm_results]).T
        elif score_format == "profile":
            bm_profile_storages = np.array(count_table.calculateProfileBindingModeAlphas())
            input_desc = []
            for bm_storage_arr in bm_profile_storages:
                bm_desc = []
                for item in bm_storage_arr:
                    bm_seq_desc = np.hstack([self.__create_numpy(item.getFirst()),
                                             self.__create_numpy(item.getSecond())])
                    if profile_aggregate is not None:
                        bm_seq_desc = self.__profile_aggr[profile_aggregate](bm_seq_desc)
                    bm_desc.append(bm_seq_desc)
                input_desc.append(bm_desc)
            result = np.array(input_desc)
        else:
            raise Exception(f"{score_format} : undefined scoring format for bindingModeScores() method.")

        return result
