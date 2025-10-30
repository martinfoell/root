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
    def define_rdf(self, num_of_entries=10):
        df = ROOT.RDataFrame(num_of_entries)\
            .Define("b1", "(int) rdfentry_")\
            .Define("b2", "(double) b1*b1")

        return df

    def create_file(self, num_of_entries=10):
        self.define_rdf(num_of_entries).Snapshot(
            self.tree_name, self.file_name1)

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
        self.create_file()

        try:
            df = ROOT.RDataFrame(self.tree_name, self.file_name1)

            gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
                df,
                batch_size=3,
                target="b2",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                load_eager = True
            )

            results_x_train = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]            
            results_x_val = [6.0, 7.0, 8.0, 9.0]            
            results_y_train = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]
            results_y_val = [36.0, 49.0, 64.0, 81.0]            

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)            
 
            for _ in range(self.n_val_batch):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())
            
            for _ in range(self.n_train_batch):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            x, y = next(val_iter)
            self.assertTrue(x.shape == (self.val_remainder, 1))
            self.assertTrue(y.shape == (self.val_remainder, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())

            flat_x_train = [
                x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [
                y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(results_x_train, flat_x_train)
            self.assertEqual(results_x_val, flat_x_val)
            self.assertEqual(results_y_train, flat_y_train)
            self.assertEqual(results_y_val, flat_y_val)

            self.teardown_file(self.file_name1)

        except:
            self.teardown_file(self.file_name1)
            raise

if __name__ == '__main__':
    unittest.main()
