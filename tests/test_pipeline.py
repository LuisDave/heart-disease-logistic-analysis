from src.data import clean_data, load_raw_data
from src.modeling import fit_logit


def test_clean_data_has_expected_unique_rows_and_binary_target():
    raw = load_raw_data()
    clean, audit = clean_data(raw)
    assert len(clean) == 302
    assert clean.duplicated().sum() == 0
    assert sorted(clean["target"].unique().tolist()) == [0, 1]
    assert int(audit.loc[audit["metric"] == "duplicates_removed", "value"].iloc[0]) == 723


def test_full_logit_converges():
    clean, _ = clean_data(load_raw_data())
    model = fit_logit(clean)
    assert model.mle_retvals.get("converged", False)
    assert 0 < model.prsquared < 1
