import pytest
import urllib3

from neural_iap_data.datasets.sgdml import MD17Dataset, MD22Dataset, RevisedMD17Dataset


@pytest.mark.parametrize("dataset_cls", [MD17Dataset, MD22Dataset, RevisedMD17Dataset])
def test_404_error(dataset_cls):
    """Test that the MD17 dataset raises a 404 error if the URL is incorrect."""
    urls = dataset_cls.urls.values()
    for url in urls:
        # Since the url contains very large files, we need to use a head request
        # to check if the url is valid.
        http = urllib3.PoolManager()
        r = http.request("HEAD", url)
        assert r.status == 200, f"URL {url} is not valid."
