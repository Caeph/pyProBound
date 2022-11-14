import numpy as np
import pandas as pd
import os
import jnius_config
import io

jardir = os.path.split(os.path.realpath(__file__))[0] + "/pyProBound"

jnius_config.add_classpath(f"{jardir}/ProBound-jar-with-dependencies.jar")
from jnius import autoclass

Locale = autoclass("java.util.Locale")
Locale.setDefault(Locale.ENGLISH)

generalSchemaFile = f"{jardir}/schema.general.json"
Toolbox = autoclass("proBoundTools.Toolbox")
Javalist = autoclass("java.util.ArrayList")


class ProBoundModel:
    """
    Class representing a ProBound model.
    """

    def __create_java_arraylist(self, iterable, throw_for_none=True):
        javalist = Javalist()
        if (iterable is None) and throw_for_none:
            raise Exception("Trying to iterate over None.")
        elif iterable is None:
            return javalist
        for item in iterable:
            javalist.add(item)

        return javalist

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
        modifications = self.__create_java_arraylist(modifications, throw_for_none=False)
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
        return output

    def __sequences_to_temp_TXT(self, sequences):
        filename = "sequences.tmp"
        np.savetxt(filename, sequences, fmt="%s")
        return filename

    __profile_aggr = {
        "sum": lambda desc: np.sum(desc, axis=-1),
        "mean": lambda desc: np.mean(desc, axis=-1),
        "max": lambda desc: np.max(desc, axis=-1),
        "forward": lambda desc: desc[:, :, :, 0],
    }

    def __score_binding_mode_scores_same_size(self,
                                              sequences,
                                              modifications,
                                              score_format="sum",
                                              profile_aggregate=None,
                                              uselist=False,
                                              ):
        tmp_filename = self.__sequences_to_temp_TXT(sequences)
        self.t.inputTXT(tmp_filename)
        tmp_scoresfile = "scores.tmp"
        self.t.bindingModeScores(tmp_scoresfile, score_format, modifications)

        result = None
        if score_format == "profile":
            # get number of binding motifs
            with open(tmp_scoresfile) as f:
               cols = f.readline().count("\t")
            bms = cols // 2
            result = np.ones((len(sequences), bms, len(sequences[0])))

            #s = io.BytesIO(open(tmp_scoresfile, 'rb').read().replace(b'\t', b','))
            #scores = np.genfromtxt(s, dtype=float, delimiter=',')[:, 1:]



            #scores = np.reshape(scores, (-1, bms, 2, slides))
            #result = np.transpose(scores, axes=[0, 1, 3, 2])

            #scores = pd.read_csv(tmp_scoresfile, sep="\t", header=None)
            #cols = list(scores.columns)
            #forw, rev = [], []
            #for i, c in enumerate(cols[1:]):
            #    bm, r = i // 2, i % 2
            #    if r == 0:
            #        forw.append(np.vstack(scores[c].str.split(",")).astype(float))
            #    else:
            #        rev.append(np.vstack(scores[c].str.split(",")).astype(float))
            #result = np.stack([forw, rev])
            #result = np.transpose(result, axes=[2, 1, 3, 0])

            #if profile_aggregate is not None:
            #    result = self.__profile_aggr[profile_aggregate](result)
        elif score_format in {"max", "sum", "mean"}:
            scores = pd.read_csv(tmp_scoresfile, sep="\t", header=None)
            cols = list(scores.columns)
            result = scores[cols[1:]].values
        else:
            # will never occur - this will be caught by the Java program
            raise Exception(f"Score format {score_format} not recognized.")

        os.remove(tmp_filename)
        # os.remove(tmp_scoresfile)
        return result

#bases = list("ACGT")
#seqs = ["".join(np.random.choice(bases, size=20)) for i in range(1000)]

#model = ProBoundModel("test_input/fit.sox2.json", fitjson=True)
# model.remove_binding_mode(0)
#result = model.score_binding_mode_scores(seqs[0:5],
                                        # score_format="profile",
                                        # profile_aggregate=None,
                                        # )
#print(result)
