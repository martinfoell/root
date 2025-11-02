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
        self.define_rdf(1, 10, num_of_entries).Snapshot(
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
        self.create_file1()
        self.create_file2()

        try:
            df1 = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df2 = ROOT.RDataFrame(self.tree_name, self.file_name2)            

            gen_train, gen_validation = ROOT.TMVA.Experimental.NumPyDataLoader(
                [df1, df2],
                batch_size=3,
                target="target",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False
            )

            results_x_train = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
            results_x_val = [6.0, 7.0, 8.0, 9.0, 16.0, 17.0, 18.0, 19.0]            
            results_y_train = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            results_y_val = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)            

            for _ in range(gen_validation.number_of_batches - 1):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            x, y = next(val_iter)
            self.assertTrue(x.shape == (2, 1))
            self.assertTrue(y.shape == (2, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())
            
            for _ in range(gen_train.number_of_batches):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())


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
            self.teardown_file(self.file_name2)            

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)            
            raise
    
    def test02_each_element_is_generated_shuffled(self):
        self.create_file1()
        self.create_file2()

        try:
            df1 = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df2 = ROOT.RDataFrame(self.tree_name, self.file_name2)            

            gen_train, gen_validation = ROOT.TMVA.Experimental.NumPyDataLoader(
                [df1, df2],
                batch_size=3,
                target="target",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False
            )

            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)            

            for _ in range(gen_validation.number_of_batches - 1):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

            x, y = next(val_iter)
            self.assertTrue(x.shape == (2, 1))
            self.assertTrue(y.shape == (2, 1))
            collected_x_val.append(x.tolist())
            collected_y_val.append(y.tolist())
            
            for _ in range(gen_train.number_of_batches):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())

            flat_x_train = [
                x for xl in collected_x_train for xs in xl for x in xs]
            flat_x_val = [x for xl in collected_x_val for xs in xl for x in xs]
            flat_y_train = [
                y for yl in collected_y_train for ys in yl for y in ys]
            flat_y_val = [y for yl in collected_y_val for ys in yl for y in ys]

            self.assertEqual(len(flat_x_train), 12)
            self.assertEqual(len(flat_x_val), 8)
            self.assertEqual(len(flat_y_train), 12)
            self.assertEqual(len(flat_y_val), 8)

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)            

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)            
            raise

    def test04_dropping_remainder(self):
        self.create_file1()
        self.create_file2()

        try:
            df1 = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df2 = ROOT.RDataFrame(self.tree_name, self.file_name2)            

            gen_train, gen_validation = ROOT.TMVA.Experimental.NumPyDataLoader(
                [df1, df2],
                batch_size=3,
                target="target",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=True
            )

            collected_x = []
            collected_y = []

            for x, y in gen_train:
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x.append(x)
                collected_y.append(y)

            for x, y in gen_validation:
                self.assertTrue(x.shape == (3, 1))
                self.assertTrue(y.shape == (3, 1))
                collected_x.append(x)
                collected_y.append(y)

            self.assertEqual(len(collected_x), 6)
            self.assertEqual(len(collected_y), 6)

            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)            

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)            
            raise

    def test10_two_epochs_unshuffled(self):
        self.create_file1()
        self.create_file2()

        try:
            df1 = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df2 = ROOT.RDataFrame(self.tree_name, self.file_name2)            

            gen_train, gen_validation = ROOT.TMVA.Experimental.NumPyDataLoader(
                [df1, df2],
                batch_size=3,
                target="target",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False
            )

            both_epochs_collected_x_val = []
            both_epochs_collected_y_val = []

            for _ in range(2):
                collected_x_train = []
                collected_x_val = []
                collected_y_train = []
                collected_y_val = []

                iter_train = iter(gen_train)
                iter_val = iter(gen_validation)

                for _ in range(gen_validation.number_of_batches - 1):
                    x, y = next(iter_val)
                    self.assertTrue(x.shape == (3, 1))
                    self.assertTrue(y.shape == (3, 1))
                    collected_x_val.append(x.tolist())
                    collected_y_val.append(y.tolist())

                x, y = next(iter_val)
                self.assertTrue(x.shape == (2, 1))
                self.assertTrue(y.shape == (2, 1))
                collected_x_val.append(x.tolist())
                collected_y_val.append(y.tolist())

                for _ in range(gen_train.number_of_batches):
                    x, y = next(iter_train)
                    self.assertTrue(x.shape == (3, 1))
                    self.assertTrue(y.shape == (3, 1))
                    collected_x_train.append(x.tolist())
                    collected_y_train.append(y.tolist())

                flat_x_train = {
                    x for xl in collected_x_train for xs in xl for x in xs}
                flat_x_val = {
                    x for xl in collected_x_val for xs in xl for x in xs}
                flat_y_train = {
                    y for yl in collected_y_train for ys in yl for y in ys}
                flat_y_val = {
                    y for yl in collected_y_val for ys in yl for y in ys}

                both_epochs_collected_x_val.append(collected_x_val)
                both_epochs_collected_y_val.append(collected_y_val)

            self.assertEqual(
                both_epochs_collected_x_val[0], both_epochs_collected_x_val[1])
            self.assertEqual(
                both_epochs_collected_y_val[0], both_epochs_collected_y_val[1])
        finally:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)            

if __name__ == '__main__':
    unittest.main()
