from metrics.benchmark import hallucination_rate, protein_recall
from PipelineB.agent1_discovery import Agent1DiscoveryResult
from PipelineB.opentargets_client import GeneAssociation


def _agent1_result(disease1_genes, disease2_genes):
    return Agent1DiscoveryResult(
        disease1="Type 2 Diabetes Mellitus",
        disease2="Alzheimer's Disease",
        disease1_efo_id="EFO_0001360",
        disease2_efo_id="EFO_0000249",
        disease1_targets=[GeneAssociation(g, f"ENSG_{g}", None, 0.5) for g in disease1_genes],
        disease2_targets=[GeneAssociation(g, f"ENSG_{g}", None, 0.5) for g in disease2_genes],
        citations={},
    )


def test_protein_recall_counts_overlap_with_agent1_genes():
    agent1_result = _agent1_result(["TCF7L2", "INS"], ["APP", "APOE"])
    cited = ["TCF7L2", "APP", "MADE-UP1"]
    # 2 of Agent 1's 4 genes (TCF7L2, APP) were also cited by Pipeline A.
    assert protein_recall(cited, agent1_result) == 0.5


def test_protein_recall_is_case_insensitive():
    agent1_result = _agent1_result(["TCF7L2"], [])
    assert protein_recall(["tcf7l2"], agent1_result) == 1.0


def test_protein_recall_zero_when_agent1_found_nothing():
    agent1_result = _agent1_result([], [])
    assert protein_recall(["TCF7L2"], agent1_result) == 0.0


def test_hallucination_rate_flags_symbols_outside_known_set():
    known = {"TCF7L2", "APP", "APOE"}
    cited = ["TCF7L2", "APP", "MADE-UP1", "MADE-UP2"]
    # 2 of 4 cited symbols aren't real genes.
    assert hallucination_rate(cited, known) == 0.5


def test_hallucination_rate_zero_when_nothing_cited():
    assert hallucination_rate([], {"TCF7L2"}) == 0.0


def test_hallucination_rate_zero_when_everything_real():
    known = {"TCF7L2", "APP"}
    assert hallucination_rate(["TCF7L2", "APP"], known) == 0.0
