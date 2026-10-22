import unittest

from s1.vehicles import Vehicle, VehicleRegistry


class TestRegistryRebind(unittest.TestCase):
    def test_rebind_preserves_duplicate_with_ocr_inflight(self):
        registry = VehicleRegistry()

        duplicate = Vehicle(
            uid="ocr-pending",
            track_id=7,
            first_ts=100.0,
        )
        duplicate.plate_inflight = True

        recovered = Vehicle(
            uid="recovered",
            track_id=9,
            first_ts=101.0,
        )

        registry.by_uid[duplicate.uid] = duplicate
        registry.by_uid[recovered.uid] = recovered
        registry.by_track[7] = duplicate
        registry.by_track[9] = recovered

        registry.rebind(7, recovered)

        self.assertIs(registry.by_track[7], recovered)
        self.assertIs(
            registry.by_uid.get(duplicate.uid),
            duplicate,
            "An in-flight OCR result must still find its original vehicle",
        )


if __name__ == "__main__":
    unittest.main()
