import logging
import re
from pathlib import Path
from typing import Callable, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from dotenv import dotenv_values
from nltk.tokenize import word_tokenize
from rich.console import Console
from scipy.spatial import distance
from sklearn import tree
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

ENV_FILE = dotenv_values(".env")
OUTPUT_DIR = cast(str, ENV_FILE.get("OUTPUT_DIR", "outputs"))
FORMAT = "[%(levelname)s] %(name)s.%(funcName)s - %(message)s"

logging.basicConfig(format=FORMAT)

console = Console()


################ UTILITIES


def get_logger(
    name=__name__,
    log_level=logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # create formatter
    formatter = logging.Formatter(FORMAT)

    if log_file is not None:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


log = get_logger(__name__)


def check_make_dirs(
    paths: str | Path | list[str | Path],
    verbose: bool = True,
    isdir: bool = False,
) -> None:
    """Create base directories for given paths if they do not exist.

    Parameters
    ----------
    paths: List[str] | str
        A path or list of paths for which to check the basedirectories
    verbose: bool, default=True
        Whether to log the output path
    isdir: bool, default=False
        Treats given path(s) as diretory instead of only checking the basedir.
    """

    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        if isdir and path != "" and not Path(path).exists():
            Path(path).mkdir(parents=True)
        elif Path(path).parent != "" and not Path(path).parent.exists():
            Path(path).parent.mkdir(parents=True)
        if verbose:
            log.info(f"Output path: {path}")


################ LOADING FUNCTIONS


def get_story_basename(story: str) -> str:
    """Returns basename of story (e.g. monkey1 -> monkey)"""
    if story.startswith("monkey"):
        story = "monkey"
    elif story.startswith("pieman"):
        story = "pieman"
    return story


def get_story_dir(story: str) -> Path:
    """Returns the dirname in Michelmann dataset for given story."""
    story = get_story_basename(story)
    base_dir = str(ENV_FILE.get("GPT_EVENT_SHARE_DIR", "data/datasets/GPT_event_share"))
    if not Path(base_dir).exists():
        raise FileNotFoundError(
            f"Cannot find data in {base_dir}. Please download the data from"
            f" https://zenodo.org/records/10055827 , unzip as GPT_event_share into data/datasets/"
            " OR set the GPT_EVENT_SHARE_DIR in the '.env' file to the path of the data."
        )
    if story == "pieman2":
        story = "pieman"
    return Path(base_dir, story.capitalize(), "sourcedata")


def get_plain_text(story: str) -> str:
    """Return str with clean text of story."""
    story = get_story_basename(story)

    story_dir = get_story_dir(story).parent
    with open(
        Path(story_dir, f"{story.capitalize()}Clean.txt"), encoding="utf-8-sig"
    ) as f_in:
        data = f_in.read()
    # there shouldn't be any \\ in the text anymore
    text_plain = data.replace('\\"', "")
    text_plain = text_plain.replace("...", ".")
    return text_plain


def load_words_and_times(story: str):
    """Adapted from Michelmann et al. 2025"""
    # % read in the words and word on-/offsets and parse them
    # pieman words are called words_b, not all tunnel words are aligned
    # tunnel needs to be extensively fixed

    story = get_story_basename(story)

    story_dir = get_story_dir(story)

    # load the words
    if story == "pieman":
        words = sio.loadmat(Path(story_dir, "words.mat"))["words_b"]
    else:
        words = sio.loadmat(Path(story_dir, "words.mat"))["words"]

    # load plain text
    text = get_plain_text(story)

    # somehow this is an array inside of an array
    words_new = []
    for i in range(len(words)):
        words_new.append(words[i][0])
    words = []
    for i in range(len(words_new)):
        words.append(str(words_new[i][0]))

    # load the word onsets and offsets
    w_onsets = sio.loadmat(Path(story_dir, "word_onsets.mat"))
    w_offsets = sio.loadmat(Path(story_dir, "word_offsets.mat"))
    if story.startswith("pieman"):
        w_onsets = w_onsets["onsets_b"]
        w_offsets = w_offsets["offsets_b"]
    else:
        w_onsets = w_onsets["onsets"]
        w_offsets = w_offsets["offsets"]
    # %
    # find problems in the story:
    # First tokenize the text into words
    words_compare = word_tokenize(re.sub(r"[^a-zA-Z0-9 ]", "", "".join(text)))
    # fix wanna gonna that are tokenized into 2 words
    for idx, w in enumerate(words_compare):
        if w.lower() == "wan":
            words_compare[idx] = "wanna"
        elif w.lower() == "gon":
            words_compare[idx] = "gonna"
    while "na" in words_compare:
        words_compare.remove("na")

    # %% run this to fix the aligned words (there are missing words in tunnel). Only allow for one word at the time to be skipped!
    index = 0
    # if a fix has already been done, we don't allow more skips!
    one_fix = True
    for w in words_compare:
        if not (w.lower().replace("'", "") == words[index].lower().replace("'", "")):
            #    print(index)
            #   print(words[index])
            if not one_fix:
                break
            # try to continue by inserting the missin word in the word list...
            tmp = words[0:index]
            tmp.append(w)
            tmp = tmp + words[index:]
            words = tmp
            # and insert a nan into the onset..
            tmp = np.concatenate(
                (w_onsets[0:index,], np.nan * np.empty((1, 1)), w_onsets[index:,]),
                axis=0,
            )
            w_onsets = tmp
            # ... and offset list
            tmp = np.concatenate(
                (w_offsets[0:index,], np.nan * np.empty((1, 1)), w_offsets[index:,]),
                axis=0,
            )
            w_offsets = tmp
            one_fix = False
        else:
            # if nothing had to be fixed on this iteration, we can do another fix later
            one_fix = True
            # break
        index = index + 1
    # now there should be the same number of words in the story and in the
    # word list with corresponding onsets.
    assert len(words_compare) == len(words)
    # return the words in the story and the onsets
    return words, w_onsets, w_offsets


def get_sentence_bounds(
    story: str,
    sentences: list[str],
    words: list[str],
    w_onsets: np.ndarray,
    w_offsets: np.ndarray,
):
    """Adapted from Michelmann et al. 2025


    Returns
    -------
    sentence_bounds : np.ndarray
        The midpoint (in ms) between the end of the previous sentence and the next sentence.
        Return n-1 entries, n = number of sentences.
    """

    index = 0
    onset_list = []
    offset_list = []
    wordsum = 0
    word_ends = []
    word_start = []
    # loop through all the sentences
    for idx_sent, s in enumerate(sentences):
        if story == "tunnel":  # fix annoying compound nouns in Tunnel
            s = s.replace("-", "")
        # get the word_list for the sentence:
        tmp_words = word_tokenize(re.sub(r"[^a-zA-Z0-9 ]", "", "".join(s)))
        # fix wanna gonna ----
        for idx, w in enumerate(tmp_words):
            if w.lower() == "wan":
                tmp_words[idx] = "wanna"
            elif w.lower() == "gon":
                tmp_words[idx] = "gonna"
        while "na" in tmp_words:
            tmp_words.remove("na")
        # ------------
        # add words to the sum
        wordsum = wordsum + len(tmp_words)
        # DeBUGGING w_index = s.casefold().find(words[index].casefold())
        # add the indexed word's onset to the list
        onset_list.append(w_onsets[index])

        # FIX NAN problems in Tunnel!If there is a nan, we use the the next onset
        if story == "tunnel":
            tmpwin = 0
            while np.isnan(onset_list[-1]):
                tmpwin = tmpwin + 1
                onset_list[-1] = w_onsets[index + tmpwin]
        word_start.append(words[index])
        # s = s[index+len(words[index])+1:]
        index = index + 1  # move the word index forward
        l_index = 0
        # as long as we are in this sentence (<wordsum) and the word can be found
        while l_index >= 0 and index < wordsum:
            # extra check to make sure we don't exceed the word-list
            if index < len(words) - 1:
                # find the last occurence of this word in the sentence
                l_index = s.casefold().rfind(words[index].casefold())
                # if the word is not found, try to fix it!
                # this may be because of im, ive and id
                if l_index == -1:
                    l_index = (
                        s.casefold().replace("'", "").rfind(words[index].casefold())
                    )
                # could be nan...
                if l_index == -1:
                    print(f"{idx_sent=} {index=}")
            # keep increasing the index
            index = index + 1

        #   print(words[index])
        # index = index -1
        word_ends.append(words[index - 1])
        offset_list.append(w_offsets[index - 1])
        # FIX NAN problems in Tunnel!
        if story == "tunnel":
            tmpwin = 0
            # if there is a nan in Tunnel, we appen the previous offset
            while np.isnan(offset_list[-1]):
                tmpwin = tmpwin + 1
                offset_list[-1] = w_offsets[index - 1 - tmpwin]

        wordsum = index
    # we should have arrived at the end
    sentence_bounds = (np.array(onset_list[1:]) + np.array(offset_list[:-1])) / 2
    # gkp: this assert checks whether there exists at least one pair in which
    # offset_list is smaller than onset_list.
    # pretty sure that is not what is wanted here, but leaving it until
    # I check the logic of the code thoroughly
    assert all(offset_list[:-1] < onset_list[1:])  # type: ignore
    return sentence_bounds, onset_list, offset_list, word_start, word_ends


def load_consensus_bounds_ms(story: str) -> np.ndarray:
    """Returns consensus boundaries as milliseconds from beginning."""

    filepostfix = ""
    if story == "pieman1" or story == "pieman":
        filepostfix = "Run1"
    elif story == "pieman2":
        filepostfix = "Run2"

    story = get_story_basename(story)

    assert story in ["pieman", "tunnel", "monkey"], "Invalid story"

    story_path = get_story_dir(story)
    with open(
        Path(story_path, f"{story.capitalize()}Bounds{filepostfix}.txt"),
        "r",
        encoding="utf-8-sig",
    ) as f_in:
        bounds_ms = np.array(list(f_in.read().splitlines()), dtype=int)
    return bounds_ms


def load_story_df(story: str, n_clauses: int | None = None) -> pd.DataFrame:
    story = get_story_basename(story)
    story_df = pd.read_csv(Path("data", "stories", f"{story}.csv"))
    if n_clauses is not None:
        story_df = story_df.iloc[:n_clauses]
    return story_df


def load_consensus_boundaries(
    story: str,
    sentences: list[str] | None = None,
    post: bool = False,
    verbose: bool = False,
    narrator_pattern: str = "<[a-zA-Z 1-9]*> ",
) -> np.ndarray:
    """Returns sentences with consensus boundaries from Michelmann et al. 2025.
    Adapted from Michelmann et al. 2025.

    Parameters
    ----------
    story : {"pieman", "pieman2", "monkey", "tunnel"}
        Story identifier.
    sentences : list of str
        A list containg N sentences, has to contain the same words as specified by the
        story, can also input the story split up into clauses, or other units.
    post : bool, default=False
        If True, boundary is always marked after the sentence it occured in. If False,
         the boundary is marked to the closest sentence start/end (e.g. if the boundary
          occured in the first half of the sentence, the boundary is marked before the
          sentence).
    narrator_pattern : str, default="<[a-zA-Z 1-9]> "
        Pattern based on which narrator tokens will be removed from the text.

    Returns
    -------
    boundaries : np.ndarray, shape = (N), dtype = int
        A vector of length N, with 0 indicating an absence of and 1 indicating
         the presence of a boundary AFTER the sentence at the index.
    """

    # 1. load human bound timings
    bounds_ms = load_consensus_bounds_ms(story)

    # 2. get the word timings
    words, w_onsets, w_offsets = load_words_and_times(story)

    if sentences is None:
        sentences = load_story_df(story)["clause"].to_list()

    # 3. remove the narrator markings
    processed_sentences = list()
    for sent in sentences:
        m = re.search(narrator_pattern, sent)
        if m is not None:
            processed_sentences.append(f"{sent[: m.start()]}{sent[m.end() :]}")
        else:
            processed_sentences.append(sent)

    # 4. compute the sentence timings
    (
        sentence_bounds,
        onset_list,
        offset_list,
        starting_words,
        ending_words,
    ) = get_sentence_bounds(story, processed_sentences, words, w_onsets, w_offsets)

    boundary_vector = np.zeros(len(sentences))

    bounds_s = bounds_ms / 1000

    if post:
        for bound in bounds_s:
            boundary_vector[sum(sentence_bounds < bound)] = 1
    else:
        for bound in bounds_s:
            boundary_vector[np.absolute(sentence_bounds - bound).argmin()] = 1

    if verbose:
        console.print("\nConsensus boundary stats:", style="yellow")
        print(f"N Boundaries: {int(boundary_vector.sum())} ")

    return boundary_vector


################ CROSSVAL FUNCTIONS


def get_prerec_strict(
    responses: pd.Series | np.ndarray, ground_truth: pd.Series | np.ndarray
) -> tuple[float, float]:
    if isinstance(responses, pd.Series):
        responses = responses.astype(bool).to_numpy()
    else:
        responses = responses.astype(bool)
    if isinstance(ground_truth, pd.Series):
        ground_truth = ground_truth.astype(bool).to_numpy()
    else:
        ground_truth = ground_truth.astype(bool)

    if responses.sum() == 0:
        return 0, 0

    true_positives = responses[ground_truth].sum()
    recall = true_positives / ground_truth.sum()
    precision = true_positives / responses.sum()
    return precision, recall


def percentile_of_nd(
    samples: pd.DataFrame | np.ndarray | list, value: np.ndarray
) -> np.ndarray:
    """For typing reasons have 2 functions"""
    if isinstance(samples, list):
        samples = np.array(samples)[None, :]
    return np.count_nonzero(samples < value, axis=0) / samples.shape[0]


def bootstrap_get_estimates_1d(
    config: dict,
    sample: np.ndarray,
    sample_agg_func: Callable,
    aggregation_args: dict | None = None,
    n_dims_result: int | None = None,
) -> np.ndarray:
    n_bootstrap = config.get("n_bootstrap", 5000)
    if n_dims_result is not None:
        estimates = np.empty((n_bootstrap, n_dims_result))
    else:
        estimates = np.empty(n_bootstrap)
    for idx in tqdm(
        range(config["n_bootstrap"]),
        desc="bootstrapping",
        total=config["n_bootstrap"],
        position=config.get("bootstrap_tqdm_position"),
        leave=config.get("bootstrap_tqdm_leave", True),
    ):
        if aggregation_args is not None:
            estimates[idx] = sample_agg_func(sample, **aggregation_args)
        else:
            estimates[idx] = sample_agg_func(sample)

    return estimates


def get_pvalue(percentile: float, alternative: str) -> float:
    if alternative == "two-sided":
        pvalue = min(1 - percentile, percentile) * 2
    elif alternative == "greater":
        pvalue = 1 - percentile
    elif alternative == "less":
        pvalue = percentile
    else:
        raise ValueError(
            'config[\'alternative\'] has to be one of "two-sided", "greater", or "less"'
            f'not "{alternative}"'
        )
    return pvalue


def get_hamming_prec_rec(
    preds: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float]:
    # remove first entry
    precision, recall = get_prerec_strict(preds, labels)
    hamming = distance.hamming(labels, preds)
    return hamming, precision, recall


def fit_eval_eim(
    config: dict,
    ratings_train: np.ndarray,
    labels_train: np.ndarray,
    ratings_test: np.ndarray,
    labels_test: np.ndarray,
    print_results: bool = True,
    shuffle_preds: bool = False,
    rng: np.random.Generator | None = None,
    fold: int | None = None,
) -> tuple[float, float, float, float, float, float]:
    # sample_weight = np.ones_like(labels_train)
    # sample_weight[(ratings_train.sum(axis=1) == 0)] = 0
    sample_weight = compute_sample_weight("balanced", labels_train)

    model_kind = config.get("model_kind", "tree")
    match model_kind:
        case "tree":
            clf = tree.DecisionTreeClassifier()
        case "logistic":
            clf = LogisticRegression()
        case "linear":
            clf = LinearRegression()
        case _:
            raise ValueError(f"Invalid: {model_kind=}")

    clf = clf.fit(ratings_train, labels_train, sample_weight=sample_weight)
    predictions_test = clf.predict(ratings_test)

    if shuffle_preds:
        assert rng is not None, "rng needs to be given if shuffle_preds=True"
        if config.get("simple_shuffle"):
            predictions_test = rng.permutation(predictions_test)

        else:
            pred_idcs = np.nonzero(predictions_test)[0]
            if len(pred_idcs) != 0:
                # get inter-event distances
                inter_event_dist = np.empty((len(pred_idcs) + 1), dtype=np.int32)
                inter_event_dist[0] = (
                    pred_idcs[0] + 1
                )  # the distance between start and first event is idx + 1
                inter_event_dist[1:-1] = pred_idcs[1:] - pred_idcs[:-1]
                inter_event_dist[-1] = len(predictions_test) - pred_idcs[-1]
                # shuffle inter-event distances
                rng.shuffle(inter_event_dist)
                # convert back to indices
                inter_event_dist[0] -= 1
                predictions_test.fill(0)
                predictions_test[inter_event_dist.cumsum()[:-1]] = 1

    n_boundaries_pred = predictions_test.sum()
    n_boundaries_labels = labels_test.sum()
    hamming, precision, recall = get_hamming_prec_rec(predictions_test, labels_test)
    if any(predictions_test) and any(labels_test):
        corr = np.corrcoef(predictions_test, labels_test)[0, 1].item()
    else:
        corr = np.nan

    if print_results:
        print(f"Model N boundaries: {n_boundaries_pred}")
        print(f"Human N boundaries: {n_boundaries_labels}")
        print(f"Hamming distance: {hamming}")
        print(f"Precision: {round(precision, 2):.2f}")
        print(f"Recall: {round(recall, 2):.2f}")
        print(f"Corr: {round(corr, 2):.2f}")
        if config.get("plot_tree"):
            fig = plt.figure(figsize=(12, 9))
            tree.plot_tree(
                clf,
                max_depth=2,
                feature_names=config.get("dimensions"),
                class_names=["0", "1"],
                proportion=True,
                rounded=True,
                fontsize=12,
            )
            fold_str = "" if fold is None else f"_fold{fold}"
            plot_path = Path(OUTPUT_DIR, "plots", f"tree{fold_str}.png")
            check_make_dirs(plot_path)
            fig.savefig(plot_path)

    return hamming, precision, recall, n_boundaries_pred, n_boundaries_labels, corr


def evaluate_boundaries_kfold(
    config: dict,
    ratings: np.ndarray,
    boundaries: np.ndarray,
    verbose: bool = True,
    shuffle_preds: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    skf = KFold(n_splits=config["n_folds"])

    hammings = list()
    precisions = list()
    recalls = list()
    n_bounds_diffs = list()
    n_bounds_preds_ls = list()
    n_bounds_labels_ls = list()
    corrs = list()
    for fold_idx, (train_indices, test_indices) in enumerate(
        skf.split(ratings, boundaries)
    ):
        if verbose:
            console.print(f"\nFold {fold_idx}", style="green bold")

        fold_train_ratings = ratings[train_indices]
        fold_train_boundaries = boundaries[train_indices]
        fold_test_ratings = ratings[test_indices]
        fold_test_boundaries = boundaries[test_indices]

        hamming, precision, recall, n_bounds_preds, n_bounds_labels, corr = (
            fit_eval_eim(
                config,
                ratings_train=fold_train_ratings,
                labels_train=fold_train_boundaries,
                ratings_test=fold_test_ratings,
                labels_test=fold_test_boundaries,
                print_results=verbose,
                shuffle_preds=shuffle_preds,
                rng=rng,
                fold=fold_idx,
            )
        )
        hammings.append(hamming)
        precisions.append(precision)
        recalls.append(recall)
        n_bounds_diffs.append(abs(n_bounds_preds - n_bounds_labels))
        n_bounds_preds_ls.append(n_bounds_preds)
        n_bounds_labels_ls.append(n_bounds_labels)
        corrs.append(corr)
    return (
        hammings,
        precisions,
        recalls,
        n_bounds_diffs,
        n_bounds_preds_ls,
        n_bounds_labels_ls,
        corrs,
    )


def evaluate_boundaries_classifier_cv(config: dict):
    """Evaluates classifier on part of story, based on other part of story."""
    story = config["story"]

    # get featues of interest & consens boundaries
    shifts_df = pd.read_csv(Path(config["ratings_file"]), index_col="id")

    human_boundaries = load_consensus_boundaries(
        story=story, post=not config.get("shift_shifts_forward", False)
    )

    dimensions = config.get(
        "dimensions",
        [
            "new-agent",
            "protagonist",
            "location",
            "temporal",
            "goal",
            "causal:no_cause",
        ],
    )
    shifts = shifts_df.loc[:, dimensions].to_numpy().astype(int)

    if config.get("shift_shifts_forward"):
        shifts[:-1] = shifts[1:]
        shifts[-1] = 0

    # save joint
    shifts_df = shifts_df.loc[:, ["clause", *dimensions]]
    shifts_df["human"] = human_boundaries

    overview_path = Path(OUTPUT_DIR, f"overview_{story}.csv")
    check_make_dirs(overview_path)
    shifts_df.to_csv(overview_path)

    # exclude first event because the model has no predictions about it.
    shifts = shifts[1:]
    human_boundaries = human_boundaries[1:]

    # run kfold cross validation
    (
        hammings,
        precisions,
        recalls,
        n_bounds_diffs,
        n_bounds_preds_ls,
        n_bounds_labels_ls,
        corrs,
    ) = evaluate_boundaries_kfold(config, shifts, human_boundaries)
    result_array = np.array(
        [
            np.mean(hammings),
            np.mean(precisions),
            np.mean(recalls),
            np.mean(n_bounds_diffs),
            np.mean(n_bounds_preds_ls),
            np.mean(n_bounds_labels_ls),
            np.mean(corrs),
        ]
    )

    # bootstrap this stuff!
    def sample_kfold_boundaries(
        ratings: np.ndarray,
        boundaries: np.ndarray,
        config: dict,
        rng,
    ) -> np.ndarray:
        # compute stat with shuffled predictions
        (
            hammings,
            precisions,
            recalls,
            n_bounds_diffs,
            n_bounds_preds_ls,
            n_bounds_labels_ls,
            corrs,
        ) = evaluate_boundaries_kfold(
            config,
            ratings,
            boundaries,
            verbose=False,
            shuffle_preds=True,
            rng=rng,
        )
        return np.array(
            [
                np.mean(hammings),
                np.mean(precisions),
                np.mean(recalls),
                np.mean(n_bounds_diffs),
                np.mean(n_bounds_preds_ls),
                np.mean(n_bounds_labels_ls),
                np.mean(corrs),
            ]
        )

    rng = np.random.default_rng(config.get("bootstrap_seed"))

    estimates = bootstrap_get_estimates_1d(
        config,
        sample=shifts,
        sample_agg_func=sample_kfold_boundaries,
        aggregation_args=dict(boundaries=human_boundaries, config=config, rng=rng),
        n_dims_result=7,
    )

    percentiles = percentile_of_nd(estimates, result_array)
    alternative = config.get("alternative", "two-sided")
    pval_h = get_pvalue(percentiles[0], alternative)
    pval_p = get_pvalue(percentiles[1], alternative)
    pval_r = get_pvalue(percentiles[2], alternative)
    pval_c = get_pvalue(percentiles[6], alternative)

    estimate_means = estimates.mean(axis=0)

    bootstrap_mean_h = round(estimate_means[0].item(), 2)
    bootstrap_mean_p = round(estimate_means[1].item(), 2)
    bootstrap_mean_r = round(estimate_means[2].item(), 2)
    bootstrap_mean_c = round(estimate_means[6].item(), 2)

    q25s = np.percentile(estimates, 2.5, axis=0)
    q25_h = round(q25s[0], 3)
    q25_p = round(q25s[1], 3)
    q25_r = round(q25s[2], 3)
    q25_c = round(q25s[6], 3)

    q75s = np.percentile(estimates, 97.5, axis=0)
    q75_h = round(q75s[0], 3)
    q75_p = round(q75s[1], 3)
    q75_r = round(q75s[2], 3)
    q75_c = round(q75s[6], 3)

    console.print("\nOverall Result", style="green bold")
    print(
        f"Avg Hamming    : {round(np.mean(hammings).item(), 2):.2f}"
        f" [{q25_h:.3f} {q75_h:.3f}]"
        f" | Boostrap: {bootstrap_mean_h:.2f}"
        f" |  p = {pval_h:.4f}"
    )
    print(
        f"Avg Precision  : {round(np.mean(precisions).item(), 2):.2f}"
        f" [{q25_p:.3f} {q75_p:.3f}]"
        f" | Boostrap: {bootstrap_mean_p:.2f}"
        f" | p = {pval_p:.4f}"
    )
    print(
        f"Avg Recall     : {round(np.mean(recalls).item(), 2):.2f}"
        f" [{q25_r:.3f} {q75_r:.3f}]"
        f" | Boostrap: {bootstrap_mean_r:.2f}"
        f" | p = {pval_r:.4f}"
    )
    print(
        f"Correlation     : {round(np.mean(corrs).item(), 2):.2f}"
        f" [{q25_c:.3f} {q75_c:.3f}]"
        f" | Boostrap: {bootstrap_mean_c:.2f}"
        f" | p = {pval_c:.4f}"
    )
    print(
        f"N preds: {round(result_array[4].item(), 2)} N bounds: {round(result_array[5].item(), 2)}"
    )


if __name__ == "__main__":
    config = {
        "story": "pieman",  # tunnel | pieman | pieman2
        "ratings_file": "data/ratings/pieman_joint.csv",
        "model_kind": "tree",
        "plot_tree": True,
        "simple_shuffle": False,
        "dimensions": [
            "new-agent",
            "protagonist",
            "location",
            "temporal",
            "goal",
            "causal:no_cause",
        ],
        # k_fold
        "n_folds": 5,
        # bootstrapping
        "bootstrap_seed": 1234,
        "n_bootstrap": 5000,
    }
    evaluate_boundaries_classifier_cv(config)
