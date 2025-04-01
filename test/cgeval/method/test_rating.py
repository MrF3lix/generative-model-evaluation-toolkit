from cgeval.rating import Ratings, Observation

observation_a_w_o = Observation(
    id="1",
    input=1, 
    output="Test Sentence", 
    oracle=1, 
    metric=1
)

observation_a_wo_o = Observation(
    id="1",
    input=1, 
    output="Test Sentence", 
    oracle=None, 
    metric=1
)


def test_init_ratings():
    labels = [0, 1]
    observations = [observation_a_w_o, observation_a_wo_o]

    ratings = Ratings(labels, observations)
    
    assert len(ratings.get_metric_ratings()) == 2
    assert len(ratings.get_oracle_ratings()) == 1


def test_compute_mixture_matrix():
    labels = [0, 1]
    observations = [observation_a_w_o, observation_a_wo_o]

    ratings = Ratings(labels, observations)
    mixture_matrix = ratings.compute_mixture_matrix()

    assert len(mixture_matrix) == 2
    assert mixture_matrix[0][1][1] == 0 # TP on Label 0
    assert mixture_matrix[1][1][1] == 1 # TP on Label 1


test_compute_mixture_matrix()