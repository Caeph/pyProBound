import jpype
import jpype.imports
import atexit
import numpy as np
import pandas as pd
import os

current = os.path.split(os.path.realpath(__file__))[0]
jardir = f"{current}"
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
        "sum": lambda desc: np.sum(desc, axis=-1),
        "mean": lambda desc: np.mean(desc, axis=-1),
        "max": lambda desc: np.max(desc, axis=-1),
        "forward": lambda desc: desc[:, :, :, 0],
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
                if neither motifcentral nor fitjson are set, a valid model json is expected.
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

    def select_binding_mode(self, bindingMode, clean=False):
        """
        Selects binding mode to use in model. By default, all are used.
        :param bindingMode: integer identifier for the binding mode
        :param clean: removes all other binding modes, interactions, enrichment models
        """
        if clean:
            self.t.selectAndCleanBindingMode(bindingMode)
        else:
            self.t.selectBindingMode(bindingMode)

    def remove_binding_mode(self, bindingMode):
        """
        Removes a binding mode from the model.
        :param bindingMode: integer identifier for the binding mode
        """
        self.t.removeBindingMode(bindingMode)

    def set_mismatch_gauge(self):
        """
        Imposes the mismatch gauge on the binding modes, meaning the top sequence has score zero.
        """
        self.t.setMismatchGauge()

    def write_model(self, filename):
        """
        Write model to a filename
        :param filename: path to the file 
        """
        self.t.writeModel(filename)

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
        # if score format is profile --  list of items for each sequence:
        #       model_binding_modes X slides X 2(forward, reverse)
        indexes, results = [], []
        for len_value, seq_size_group in self.__group_sequences(sequences):
            r = self.__score_binding_mode_scores_same_size(seq_size_group.values,
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
                                              uselist=False,
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
            bm_profile_storages = count_table.calculateProfileBindingModeAlphas()
            bm_profile_storages = pd.DataFrame(bm_profile_storages)

            init = np.zeros((
                len(sequences), 
                len(bm_profile_storages.columns),
                len(sequences[0]),
                2
            ))

            for col in bm_profile_storages.columns:
                # this part takes a lot of time -- jpype...
                firsts = bm_profile_storages[col].apply(lambda stor: stor.getFirst()) # sequences X slides
                seconds = bm_profile_storages[col].apply(lambda stor: stor.getSecond()) # sequences X slides

                slides = len(firsts[0])
                for i in range(len(sequences)):
                    init[i, col, :slides, 0] = firsts[i]
                    init[i, col, :slides, 1] = seconds[i]

            result = init[:, :, :slides, :]

            if profile_aggregate is not None:
                result = self.__profile_aggr[profile_aggregate](result)
                # todo clip excess at the end -- sequence size >= no_slides
        else:
            raise Exception(f"{score_format} : undefined scoring format for bindingModeScores() method.")

        return result

    def get_best_binding_windows(self, sequence, binding_mode=0, no_best=None, padding=None):
        """
        Returns a numpy array of the best binding window sequences with position and binding score.
        :string sequence: sequence to probe
        :param binding_mode: integer indentifier of a binding mode.
                        Multiple binding modes are not supported.
                        If your model has only a single binding mode
                        or you have already picked a binding mode for the model, leave default value.
        :param no_best: number of best examples to return with respect to padding calculation.
                        If None, return all possible.
        :param padding: minimal required padding between returned windows. Default: half of PSAM size.
                        Set 0 to ignore padding.
                        The following is iterated:
                        the best window is identified, the too-close windows are removed.
        """

        vals = self.__score_binding_mode_scores_same_size([sequence],
                                                          score_format="profile",
                                                          profile_aggregate="max"
                                                          )[0][binding_mode]
        # best = np.argsort(vals)
        windowsize = len(sequence) - len(vals) + 1
        result = []

        if no_best is None:
            no_best = len(vals)

        if padding is None:
            padding = windowsize // 2

        start_pos = np.argsort(vals)[::-1]  # starting positions in descending order
        affinities = vals[start_pos]
        window_seqs = np.array([sequence[x:x + windowsize] for x in start_pos])

        results = []
        while len(results) < no_best:
            current_startpos, current_affinity, currenct_window = start_pos[0], affinities[0], window_seqs[0]

            results.append([current_startpos, current_affinity, currenct_window])

            close_posi = np.arange(current_startpos - padding, current_startpos + padding + 1)
            close_posi = close_posi[(close_posi >= 0) & (close_posi < len(vals))]

            # at these positions there is a close one
            close_indices = np.where(np.isin(start_pos, close_posi))[0]

            mask = np.ones(start_pos.size, dtype=bool)
            mask[close_indices] = False
            start_pos, affinities, window_seqs = start_pos[mask], affinities[mask], window_seqs[mask]

            if len(start_pos) == 0:
                break

        return np.array(results)
