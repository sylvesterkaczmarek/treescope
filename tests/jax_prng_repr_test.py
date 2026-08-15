# Copyright 2026 The Treescope Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
import jax
from treescope.external import jax_support


class JaxPrngReprTest(absltest.TestCase):

  def test_faster_array_repr_large_typed_prng_keys(self):
    keys = jax.random.split(jax.random.key(0, impl="threefry2x32"), 2000)

    self.assertEqual(jax_support.faster_array_repr(keys), repr(keys))


if __name__ == "__main__":
  absltest.main()
