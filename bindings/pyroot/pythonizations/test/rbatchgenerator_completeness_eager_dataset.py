import unittest
import os
import ROOT
from ROOT import RVec
import numpy as np
from random import randrange

class RBatchGeneratorMultipleFiles(unittest.TestCase):

    file_name1 = "first_half.root"
    file_name2 = "second_half.root"
    file_name3 = "vector_columns.root"    
    tree_name = "mytree"

    # default constants
    n_train_batch = 2
    n_val_batch = 1
    val_remainder = 1

    # Helpers
    def define_rdf(self, target=1, start=0, num_of_entries=10):
        df = ROOT.RDataFrame(num_of_entries)\
            .Define("column", f"(int) {start} + rdfentry_")\
            .Define("target", f"(int) {target}")

        return df

    def create_file1(self, num_of_entries=10):
        self.define_rdf(0, 0, num_of_entries).Snapshot(
            self.tree_name, self.file_name1)

    def create_file2(self, num_of_entries=10):
        self.define_rdf(1, 20, num_of_entries).Snapshot(
            self.tree_name, self.file_name2)
        
    def create_5_entries_file(self):
        df1 = ROOT.RDataFrame(5)\
            .Define("b1", "(int) rdfentry_ + 10")\
            .Define("b2", "(double) b1 * b1")\
            .Snapshot(self.tree_name, self.file_name2)

    def create_vector_file(self, num_of_entries=10):
        df3 = ROOT.RDataFrame(10)\
             .Define("b1", "(int) rdfentry_")\
             .Define("v1", "ROOT::VecOps::RVec<int>{ b1,  b1 * 10}")\
             .Define("v2", "ROOT::VecOps::RVec<int>{ b1 * 100,  b1 * 1000}")\
             .Snapshot(self.tree_name, self.file_name3)             
    
    def teardown_file(self, file):
        os.remove(file)
    
    def test01_each_element_is_generated_unshuffled(self):    
        self.create_file1(num_of_entries=10)
        self.create_file2(num_of_entries=10)

        try:
            df1 = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df2 = ROOT.RDataFrame(self.tree_name, self.file_name2)            

            dataset_train, dataset_validation = ROOT.TMVA.Experimental.NumPyDataset(
                [df1, df2],
                target="target",
                validation_split=0.4,
                shuffle=False,
            )

            collected_x_train, collected_y_train = dataset_train
            collected_x_val, collected_y_val = dataset_validation
            
            results_x_train = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
            results_x_val = [6.0, 7.0, 8.0, 9.0, 26.0, 27.0, 28.0, 29.0]
            results_y_train = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            results_y_val = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
            
            flat_x_train = [x for xs in collected_x_train for x in xs]
            flat_x_val = [x for xs in collected_x_val for x in xs]
            flat_y_train = [x for xs in collected_y_train for x in xs]
            flat_y_val = [x for xs in collected_y_val for x in xs]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)            

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)            
            raise

    def test03_TensorFlow(self):    
        self.create_file1(num_of_entries=10)
        self.create_file2(num_of_entries=10)

        try:
            df1 = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df2 = ROOT.RDataFrame(self.tree_name, self.file_name2)            

            dataset_train, dataset_validation = ROOT.TMVA.Experimental.TensorFlowDataset(
                [df1, df2],
                target="target",
                validation_split=0.4,
                shuffle=True,
                set_seed=42,
            )

            collected_x_train, collected_y_train = dataset_train
            collected_x_val, collected_y_val = dataset_validation
            
            
            results_x_train = [6.0, 21.0, 26.0, 1.0, 9.0, 20.0, 29.0, 7.0, 25.0, 5.0, 0.0, 27.0]
            results_x_val = [2.0, 23.0, 24.0, 8.0, 22.0, 28.0, 4.0, 3.0]
            results_y_train = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
            results_y_val = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        
            flat_x_train = [x for xs in collected_x_train for x in xs]
            flat_x_val = [x for xs in collected_x_val for x in xs]
            flat_y_train = [x for xs in collected_y_train for x in xs]
            flat_y_val = [x for xs in collected_y_val for x in xs]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)            

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)            
            raise

    def test04_PyTorch(self):    
        self.create_file1(num_of_entries=10)
        self.create_file2(num_of_entries=10)

        try:
            df1 = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df2 = ROOT.RDataFrame(self.tree_name, self.file_name2)            

            dataset_train, dataset_validation = ROOT.TMVA.Experimental.PyTorchDataset(
                [df1, df2],
                target="target",
                validation_split=0.4,
                shuffle=True,
                set_seed=42,
            )
            
            collected_x_train, collected_y_train = dataset_train
            collected_x_val, collected_y_val = dataset_validation
            
            
            results_x_train = [6.0, 21.0, 26.0, 1.0, 9.0, 20.0, 29.0, 7.0, 25.0, 5.0, 0.0, 27.0]
            results_x_val = [2.0, 23.0, 24.0, 8.0, 22.0, 28.0, 4.0, 3.0]
            results_y_train = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
            results_y_val = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        
            flat_x_train = [x for xs in collected_x_train for x in xs]
            flat_x_val = [x for xs in collected_x_val for x in xs]
            flat_y_train = [x for xs in collected_y_train for x in xs]
            flat_y_val = [x for xs in collected_y_val for x in xs]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)            

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)            
            raise
        
if __name__ == '__main__':
    unittest.main()
        
