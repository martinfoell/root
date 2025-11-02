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
        self.define_rdf(1, 30, num_of_entries).Snapshot(
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
    
    def test01_undersampling_without_random_state_replacement_false_unshuffled(self):
        self.create_file1(num_of_entries=30)
        self.create_file2(num_of_entries=10)

        try:
            df1 = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df2 = ROOT.RDataFrame(self.tree_name, self.file_name2)            

            undersampler = ROOT.TMVA.Experimental.RandomUnderSampler(
                sampling_strategy = 0.5,
                random_state = 0,                
                replacement = False,
            )

            gen_train, gen_validation = ROOT.TMVA.Experimental.NumPyDataLoader(
                [df1, df2],
                batch_size=4,
                target="target",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                sampler = undersampler,
            )

            results_x_train = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0]
            results_x_val = [18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 36.0, 37.0, 38.0, 39.0]
            results_y_train = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            results_y_val = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
            
            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)            

            for _ in range(gen_train.number_of_batches - 1):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (4, 1))
                self.assertTrue(y.shape == (4, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())
            
            x, y = next(train_iter)
            self.assertTrue(x.shape == (2, 1))
            self.assertTrue(y.shape == (2, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())
            
            for _ in range(gen_validation.number_of_batches):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (4, 1))
                self.assertTrue(y.shape == (4, 1))
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

            num_train_minor = sum(flat_y_train)
            num_train_resample_major = len(flat_y_train) - num_train_minor
            alpha_train = num_train_minor / num_train_resample_major

            num_val_minor = sum(flat_y_val)
            num_val_resample_major = len(flat_y_val) - num_val_minor
            alpha_val = num_val_minor / num_val_resample_major
            
            self.assertEqual(alpha_train, 0.5)            
            self.assertEqual(alpha_val, 0.5)
            
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)            

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)            
            raise

    def test02_undersampling_with_random_state_replacement_false_shuffled(self):
        self.create_file1(num_of_entries=30)
        self.create_file2(num_of_entries=10)

        try:
            df1 = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df2 = ROOT.RDataFrame(self.tree_name, self.file_name2)            

            undersampler = ROOT.TMVA.Experimental.RandomUnderSampler(
                sampling_strategy = 0.5,
                random_state = 1,                
                replacement = False,
            )

            gen_train, gen_validation = ROOT.TMVA.Experimental.NumPyDataLoader(
                [df1, df2],
                batch_size=4,
                target="target",
                validation_split=0.4,
                shuffle=True,
                set_seed=42,
                drop_remainder=False,
                sampler=undersampler,
            )

            results_x_train = [13.0, 21.0, 35.0, 19.0, 5.0, 18.0, 23.0, 31.0, 37.0, 27.0, 36.0, 39.0, 7.0, 26.0, 1.0, 30.0, 29.0, 8.0]
            results_x_val = [17.0, 6.0, 11.0, 25.0, 16.0, 32.0, 34.0, 0.0, 33.0, 20.0, 12.0, 38.0]
            results_y_train = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            results_y_val = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
            
            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)            

            for _ in range(gen_train.number_of_batches - 1):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (4, 1))
                self.assertTrue(y.shape == (4, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())
            
            x, y = next(train_iter)
            self.assertTrue(x.shape == (2, 1))
            self.assertTrue(y.shape == (2, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())
            
            for _ in range(gen_validation.number_of_batches):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (4, 1))
                self.assertTrue(y.shape == (4, 1))
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

            num_train_minor = sum(flat_y_train)
            num_train_resample_major = len(flat_y_train) - num_train_minor
            alpha_train = num_train_minor / num_train_resample_major

            num_val_minor = sum(flat_y_val)
            num_val_resample_major = len(flat_y_val) - num_val_minor
            alpha_val = num_val_minor / num_val_resample_major
            
            self.assertEqual(alpha_train, 0.5)            
            self.assertEqual(alpha_val, 0.5)
            
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)            

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)            
            raise

    def test03_undersampling_with_random_state_replacement_true_unshuffled(self):
        self.create_file1(num_of_entries=30)
        self.create_file2(num_of_entries=10)

        try:
            df1 = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df2 = ROOT.RDataFrame(self.tree_name, self.file_name2)            

            undersampler = ROOT.TMVA.Experimental.RandomUnderSampler(
                sampling_strategy = 0.5,
                random_state = 1,                
                replacement = True,
            )

            gen_train, gen_validation = ROOT.TMVA.Experimental.NumPyDataLoader(
                [df1, df2],
                batch_size=4,
                target="target",
                validation_split=0.4,
                shuffle=False,
                drop_remainder=False,
                sampler=undersampler,
            )

            results_x_train = [7.0, 17.0, 12.0, 16.0, 0.0, 2.0, 5.0, 17.0, 2.0, 4.0, 1.0, 7.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0]
            results_x_val = [23.0, 29.0, 26.0, 29.0, 18.0, 19.0, 21.0, 29.0, 36.0, 37.0, 38.0, 39.0]
            results_y_train = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            results_y_val = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
            
            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)            

            for _ in range(gen_train.number_of_batches - 1):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (4, 1))
                self.assertTrue(y.shape == (4, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())
            
            x, y = next(train_iter)
            self.assertTrue(x.shape == (2, 1))
            self.assertTrue(y.shape == (2, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())
            
            for _ in range(gen_validation.number_of_batches):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (4, 1))
                self.assertTrue(y.shape == (4, 1))
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

            num_train_minor = sum(flat_y_train)
            num_train_resample_major = len(flat_y_train) - num_train_minor
            alpha_train = num_train_minor / num_train_resample_major

            num_val_minor = sum(flat_y_val)
            num_val_resample_major = len(flat_y_val) - num_val_minor
            alpha_val = num_val_minor / num_val_resample_major
            
            self.assertEqual(alpha_train, 0.5)            
            self.assertEqual(alpha_val, 0.5)
            
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)            

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)            
            raise
    
    def test04_undersampling_with_random_state_replacement_true_shuffled(self):
        self.create_file1(num_of_entries=30)
        self.create_file2(num_of_entries=10)

        try:
            df1 = ROOT.RDataFrame(self.tree_name, self.file_name1)
            df2 = ROOT.RDataFrame(self.tree_name, self.file_name2)            

            undersampler = ROOT.TMVA.Experimental.RandomUnderSampler(
                sampling_strategy = 0.5,
                random_state = 1,                
                replacement = True,
            )

            gen_train, gen_validation = ROOT.TMVA.Experimental.NumPyDataLoader(
                [df1, df2],
                batch_size=4,
                target="target",
                validation_split=0.4,
                shuffle=True,
                set_seed=41,
                drop_remainder=False,
                sampler=undersampler,
            )

            results_x_train = [31.0, 33.0, 0.0, 27.0, 10.0, 13.0, 23.0, 37.0, 13.0, 36.0, 27.0, 20.0, 24.0, 23.0, 15.0, 39.0, 16.0, 30.0]
            results_x_val = [35.0, 32.0, 4.0, 19.0, 19.0, 6.0, 38.0, 28.0, 34.0, 22.0, 19.0, 25.0]
            results_y_train = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]
            results_y_val = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            
            collected_x_train = []
            collected_x_val = []
            collected_y_train = []
            collected_y_val = []

            train_iter = iter(gen_train)
            val_iter = iter(gen_validation)            

            for _ in range(gen_train.number_of_batches - 1):
                x, y = next(train_iter)
                self.assertTrue(x.shape == (4, 1))
                self.assertTrue(y.shape == (4, 1))
                collected_x_train.append(x.tolist())
                collected_y_train.append(y.tolist())
            
            x, y = next(train_iter)
            self.assertTrue(x.shape == (2, 1))
            self.assertTrue(y.shape == (2, 1))
            collected_x_train.append(x.tolist())
            collected_y_train.append(y.tolist())
            
            for _ in range(gen_validation.number_of_batches):
                x, y = next(val_iter)
                self.assertTrue(x.shape == (4, 1))
                self.assertTrue(y.shape == (4, 1))
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

            num_train_minor = sum(flat_y_train)
            num_train_resample_major = len(flat_y_train) - num_train_minor
            alpha_train = num_train_minor / num_train_resample_major

            num_val_minor = sum(flat_y_val)
            num_val_resample_major = len(flat_y_val) - num_val_minor
            alpha_val = num_val_minor / num_val_resample_major
            
            self.assertEqual(alpha_train, 0.5)            
            self.assertEqual(alpha_val, 0.5)
            
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)            

        except:
            self.teardown_file(self.file_name1)
            self.teardown_file(self.file_name2)            
            raise
        
if __name__ == '__main__':
    unittest.main()
