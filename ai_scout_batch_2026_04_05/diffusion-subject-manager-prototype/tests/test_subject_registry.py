import unittest
import uuid

# For the purpose of providing a self-contained test file as requested,
# we include minimal definitions for Subject and SubjectRegistry here.
# In a real project, these would be imported from src/subject_registry.py.

class Subject:
    """
    A minimal Subject class for testing purposes, mimicking the expected structure.
    """
    def __init__(self, ID, bounding_box=None, action_queue=None, latent_token_idx=None, identity_features=None):
        if not isinstance(ID, (str, uuid.UUID, int)):
            raise TypeError("Subject ID must be a string, UUID, or integer.")
        self.ID = ID
        self.bounding_box = bounding_box if bounding_box is not None else []
        self.action_queue = action_queue if action_queue is not None else []
        self.latent_token_idx = latent_token_idx
        self.identity_features = identity_features

    def __eq__(self, other):
        if not isinstance(other, Subject):
            return NotImplemented
        return self.ID == other.ID and \
               self.bounding_box == other.bounding_box and \
               self.action_queue == other.action_queue and \
               self.latent_token_idx == other.latent_token_idx and \
               self.identity_features == other.identity_features

    def __repr__(self):
        return (f"Subject(ID='{self.ID}', bounding_box={self.bounding_box}, "
                f"action_queue={self.action_queue}, latent_token_idx={self.latent_token_idx}, "
                f"identity_features={'[...]' if self.identity_features else 'None'})")

class SubjectRegistry:
    """
    A minimal SubjectRegistry class for testing purposes, mimicking the expected structure.
    """
    def __init__(self):
        self._subjects = {}

    def add_subject(self, subject: Subject):
        if not isinstance(subject, Subject):
            raise TypeError("Only Subject objects can be added to the registry.")
        if subject.ID in self._subjects:
            raise ValueError(f"Subject with ID '{subject.ID}' already exists.")
        self._subjects[subject.ID] = subject

    def remove_subject(self, subject_id):
        if subject_id not in self._subjects:
            raise ValueError(f"Subject with ID '{subject_id}' not found.")
        del self._subjects[subject_id]

    def update_subject_state(self, subject_id, **kwargs):
        if subject_id not in self._subjects:
            raise ValueError(f"Subject with ID '{subject_id}' not found for update.")
        
        subject = self._subjects[subject_id]
        
        updatable_attributes = {'bounding_box', 'action_queue', 'latent_token_idx', 'identity_features'}
        
        for key, value in kwargs.items():
            if key in updatable_attributes:
                setattr(subject, key, value)
            else:
                raise AttributeError(f"Subject attribute '{key}' is not recognized or not allowed to be updated directly via this method.")

    def get_subject_info(self, subject_id):
        return self._subjects.get(subject_id)

    def list_all_subject_ids(self):
        return list(self._subjects.keys())

    def get_num_subjects(self):
        return len(self._subjects)

    def clear(self):
        self._subjects.clear()


class TestSubjectRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = SubjectRegistry()

    def test_initialization(self):
        self.assertEqual(self.registry.get_num_subjects(), 0)
        self.assertEqual(self.registry.list_all_subject_ids(), [])

    def test_add_subject(self):
        subject_id = str(uuid.uuid4())
        subject = Subject(ID=subject_id, bounding_box=[0, 0, 10, 10], action_queue=["idle"])
        self.registry.add_subject(subject)

        self.assertEqual(self.registry.get_num_subjects(), 1)
        retrieved_subject = self.registry.get_subject_info(subject_id)
        self.assertIsNotNone(retrieved_subject)
        self.assertEqual(retrieved_subject.ID, subject_id)
        self.assertEqual(retrieved_subject.bounding_box, [0, 0, 10, 10])
        self.assertEqual(retrieved_subject.action_queue, ["idle"])
        self.assertIsNone(retrieved_subject.latent_token_idx)
        self.assertIsNone(retrieved_subject.identity_features)

    def test_add_multiple_subjects(self):
        subject_ids = [str(uuid.uuid4()) for _ in range(3)]
        subjects = [
            Subject(ID=subject_ids[0], bounding_box=[0, 0, 10, 10]),
            Subject(ID=subject_ids[1], action_queue=["walking"]),
            Subject(ID=subject_ids[2], latent_token_idx=2)
        ]

        for s in subjects:
            self.registry.add_subject(s)

        self.assertEqual(self.registry.get_num_subjects(), 3)
        self.assertCountEqual(self.registry.list_all_subject_ids(), subject_ids)

    def test_add_existing_subject_raises_error(self):
        subject_id = str(uuid.uuid4())
        subject1 = Subject(ID=subject_id)
        self.registry.add_subject(subject1)

        subject2 = Subject(ID=subject_id)
        with self.assertRaisesRegex(ValueError, f"Subject with ID '{subject_id}' already exists."):
            self.registry.add_subject(subject2)

    def test_add_non_subject_type_raises_error(self):
        with self.assertRaisesRegex(TypeError, "Only Subject objects can be added to the registry."):
            self.registry.add_subject({"ID": "dummy", "coords": [1,2,3,4]})

    def test_remove_subject(self):
        subject_id_to_remove = str(uuid.uuid4())
        subject_to_remove = Subject(ID=subject_id_to_remove)
        self.registry.add_subject(subject_to_remove)

        other_subject_id = str(uuid.uuid4())
        self.registry.add_subject(Subject(ID=other_subject_id))

        self.assertEqual(self.registry.get_num_subjects(), 2)
        self.registry.remove_subject(subject_id_to_remove)
        self.assertEqual(self.registry.get_num_subjects(), 1)
        self.assertIsNone(self.registry.get_subject_info(subject_id_to_remove))
        self.assertIsNotNone(self.registry.get_subject_info(other_subject_id))
        self.assertCountEqual(self.registry.list_all_subject_ids(), [other_subject_id])

    def test_remove_non_existent_subject_raises_error(self):
        non_existent_id = str(uuid.uuid4())
        with self.assertRaisesRegex(ValueError, f"Subject with ID '{non_existent_id}' not found."):
            self.registry.remove_subject(non_existent_id)

    def test_update_subject_state(self):
        subject_id = str(uuid.uuid4())
        subject = Subject(ID=subject_id, bounding_box=[0, 0, 10, 10], action_queue=["initial"])
        self.registry.add_subject(subject)

        new_bbox = [10, 20, 30, 40]
        new_action = ["walking", "waving"]
        new_latent_idx = 5
        new_identity_features = [0.1, 0.2, 0.3]

        self.registry.update_subject_state(
            subject_id,
            bounding_box=new_bbox,
            action_queue=new_action,
            latent_token_idx=new_latent_idx,
            identity_features=new_identity_features
        )

        updated_subject = self.registry.get_subject_info(subject_id)
        self.assertEqual(updated_subject.bounding_box, new_bbox)
        self.assertEqual(updated_subject.action_queue, new_action)
        self.assertEqual(updated_subject.latent_token_idx, new_latent_idx)
        self.assertEqual(updated_subject.identity_features, new_identity_features)

    def test_update_non_existent_subject_raises_error(self):
        non_existent_id = str(uuid.uuid4())
        with self.assertRaisesRegex(ValueError, f"Subject with ID '{non_existent_id}' not found for update."):
            self.registry.update_subject_state(non_existent_id, bounding_box=[1,1,1,1])

    def test_update_invalid_attribute_raises_error(self):
        subject_id = str(uuid.uuid4())
        self.registry.add_subject(Subject(ID=subject_id))

        with self.assertRaisesRegex(AttributeError, "Subject attribute 'invalid_attr' is not recognized or not allowed to be updated directly via this method."):
            self.registry.update_subject_state(subject_id, invalid_attr="some_value")
        
        with self.assertRaisesRegex(AttributeError, "Subject attribute 'ID' is not recognized or not allowed to be updated directly via this method."):
            self.registry.update_subject_state(subject_id, ID="new_id")

    def test_get_subject_info_returns_none_for_non_existent(self):
        self.assertIsNone(self.registry.get_subject_info(str(uuid.uuid4())))

    def test_clear_registry(self):
        self.registry.add_subject(Subject(ID=str(uuid.uuid4())))
        self.registry.add_subject(Subject(ID=str(uuid.uuid4())))
        self.assertEqual(self.registry.get_num_subjects(), 2)

        self.registry.clear()
        self.assertEqual(self.registry.get_num_subjects(), 0)
        self.assertEqual(self.registry.list_all_subject_ids(), [])

    def test_subject_id_type_validation(self):
        with self.assertRaises(TypeError):
            Subject(ID=None)
        with self.assertRaises(TypeError):
            Subject(ID={})
        
        Subject(ID="string_id")
        Subject(ID=uuid.uuid4())
        Subject(ID=123)

    def test_subject_equality(self):
        id_val = str(uuid.uuid4())
        s1 = Subject(ID=id_val, bounding_box=[0,0,1,1], action_queue=["a"], latent_token_idx=0, identity_features=[0.1])
        s2 = Subject(ID=id_val, bounding_box=[0,0,1,1], action_queue=["a"], latent_token_idx=0, identity_features=[0.1])
        s3 = Subject(ID=str(uuid.uuid4()), bounding_box=[0,0,1,1], action_queue=["a"], latent_token_idx=0, identity_features=[0.1])
        s4 = Subject(ID=id_val, bounding_box=[1,1,2,2], action_queue=["a"], latent_token_idx=0, identity_features=[0.1])

        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)
        self.assertNotEqual(s1, s4)
        self.assertNotEqual(s1, "not a subject")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)