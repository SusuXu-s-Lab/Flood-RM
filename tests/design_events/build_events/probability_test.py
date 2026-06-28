from design_events.build_events import probability
from design_events.build_events.probability import DriverDependenceModel


def test_probability_seam_exports_copula_joint_catalog_names():
    assert DriverDependenceModel is probability.DriverDependenceModel
    assert "build_joint_catalog" in probability.__all__
    assert "label_and_joint_exceedance" in probability.__all__
    assert "attach_field_preserving_realization" in probability.__all__
