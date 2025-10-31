// Author: Dante Niewenhuis, VU Amsterdam 07/2023
// Author: Kristupas Pranckietis, Vilnius University 05/2024
// Author: Nopphakorn Subsa-Ard, King Mongkut's University of Technology Thonburi (KMUTT) (TH) 08/2024
// Author: Vincenzo Eduardo Padulano, CERN 10/2024
// Author: Martin Føll, University of Oslo (UiO) & CERN 05/2025

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TMVA_RBATCHGENERATOR
#define TMVA_RBATCHGENERATOR

#include "TMVA/RTensor.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"
#include "TMVA/BatchGenerator/RChunkLoader.hxx"
#include "TMVA/BatchGenerator/RBatchLoader.hxx"
#include "TMVA/BatchGenerator/RDatasetLoader.hxx"
#include "TMVA/BatchGenerator/RSampler.hxx"
#include "TMVA/BatchGenerator/RTensorOperations.hxx"
#include "TROOT.h"

#include <cmath>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <variant>
#include <vector>
#include <chrono>

namespace TMVA {
namespace Experimental {
namespace Internal {

// clang-format off
/**
\class ROOT::TMVA::Experimental::Internal::RBatchGenerator
\ingroup tmva
\brief 

In this class, the processes of loading chunks (see RChunkLoader) and creating batches from those chunks (see RBatchLoader) are combined, allowing batches from the training and validation sets to be loaded directly from a dataset in an RDataFrame.
*/

template <typename... Args>
class RBatchGenerator {
private:
   std::vector<std::string> fCols;
   // clang-format on
   std::size_t fChunkSize;
   std::size_t fMaxChunks;
   std::size_t fBatchSize;
   std::size_t fBlockSize;
   std::size_t fNumColumns;
   std::size_t fNumDatasetCols;
   std::size_t fNumEntries;
   std::size_t fDNumEntries;
   std::size_t fSetSeed;
   std::size_t fSumVecSizes;

   std::size_t fNumDataFrames;
   std::vector<std::size_t> fDataframeEntries;   
   std::vector<std::size_t> fDataframeTrainingEntries;   
   std::vector<std::size_t> fDataframeValidationEntries;

   std::size_t fSumNumEntries; 
   std::vector<float> fMixtureWeights;
   std::vector<float> fDataframeBatchFractions;
   std::vector<std::size_t> fDataframeBatchSizes;
   ROOT::RDF::RResultPtr<std::vector<ULong64_t>> fEntries;
   float fValidationSplit;

   std::string fSampleType;
   float fSampleStrategy;
   std::unique_ptr<RChunkLoader<Args...>> fChunkLoader;
   std::unique_ptr<RDatasetLoader<Args...>> fDatasetLoader;
   std::unique_ptr<RBatchLoader> fBatchLoader;
   
   std::unique_ptr<RBatchLoader> fChunkTrainingBatchLoader;
   std::unique_ptr<RBatchLoader> fChunkValidationBatchLoader;
   
   std::unique_ptr<RBatchLoader> fTrainingBatchLoader;
   std::unique_ptr<RBatchLoader> fValidationBatchLoader;

   std::unique_ptr<RBatchLoader> fSampleTrainingBatchLoader;
   std::unique_ptr<RBatchLoader> fSampleValidationBatchLoader;
   std::unique_ptr<RTensorOperations> fTensorOperations;
   
   std::unique_ptr<RSampler> fRSampler;
   std::unique_ptr<std::thread> fLoadingThread;

   std::size_t fTrainingChunkNum;
   std::size_t fValidationChunkNum;

   std::vector<ROOT::RDF::RNode> f_rdfs;

   std::mutex fIsActiveMutex;

   bool fDropRemainder;
   bool fShuffle;
   bool fIsActive{false}; // Whether the loading thread is active
   bool fNotFiltered;
   bool fUseWholeFile;
   bool fLoadEager;
   bool fDataframeBalance{false};
   bool fQueue{true};
   bool fVector{false};
  
   bool fEpochActive{false};
   bool fTrainingEpochActive{false};
   bool fValidationEpochActive{false};

   std::size_t fNumTrainingEntries;
   std::size_t fNumValidationEntries;

   std::size_t fNumTrainingChunks;
   std::size_t fNumValidationChunks;

   std::size_t fLeftoverTrainingBatchSize;
   std::size_t fLeftoverValidationBatchSize;

   std::size_t fNumFullTrainingBatches;
   std::size_t fNumFullValidationBatches;

   std::size_t fNumLeftoverTrainingBatches;
   std::size_t fNumLeftoverValidationBatches;

   std::size_t fNumTrainingBatches;
   std::size_t fNumValidationBatches;

   TMVA::Experimental::RTensor<float> fTrainTensor;
   TMVA::Experimental::RTensor<float> fTrainChunkTensor;

   TMVA::Experimental::RTensor<float> fValidationTensor;
   TMVA::Experimental::RTensor<float> fValidationChunkTensor;

   TMVA::Experimental::RTensor<float> fTrainDatasetTensor;
   TMVA::Experimental::RTensor<float> fValidationDatasetTensor;   

   std::vector<TMVA::Experimental::RTensor<float>> fTrainingTensors;
   std::vector<TMVA::Experimental::RTensor<float>> fValidationTensors;

   std::queue<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fTrainingBatchQueue;
   std::queue<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fValidationBatchQueue;
   std::vector<std::vector<TMVA::Experimental::RTensor<float>>> fDataFrameBatchVectors;
  
public:
   RBatchGenerator(const std::vector<ROOT::RDF::RNode> &rdfs, const std::vector<float> &mixtureWeights = {}, const std::size_t chunkSize = 0,
                   const std::size_t blockSize = 0, const std::size_t batchSize = 0, const std::vector<std::string> &cols = {},
                   const std::vector<std::size_t> &vecSizes = {}, const float vecPadding = 0.0, const float validationSplit = 0.0, const std::size_t maxChunks = 0, bool shuffle = true,
                   bool dropRemainder = true, const std::size_t setSeed = 0, bool loadEager = false, std::string sampleType = "random", float sampleStrategy = 0.5)

      : f_rdfs(rdfs),
        fMixtureWeights(mixtureWeights),
        fCols(cols),
        fChunkSize(chunkSize),
        fBlockSize(blockSize),
        fBatchSize(batchSize),
        fValidationSplit(validationSplit),
        fSampleType(sampleType),
        fSampleStrategy(sampleStrategy),
        fMaxChunks(maxChunks),
        fDropRemainder(dropRemainder),
        fSetSeed(setSeed),
        fLoadEager(loadEager),
        fShuffle(shuffle),
        fNotFiltered(f_rdfs[0].GetFilterNames().empty()),
        fUseWholeFile(maxChunks == 0),
        fNumColumns(cols.size()),
        fTrainTensor({0, 0}),
        fTrainChunkTensor({0, 0}),
        fValidationTensor({0, 0}),
        fValidationChunkTensor({0, 0}),
        fTrainDatasetTensor({0, 0}),
        fValidationDatasetTensor({0, 0})
   {

      fDNumEntries = 1111;
      fRSampler = std::make_unique<RSampler>(42, fShuffle, fSetSeed, fLoadEager);
      fTensorOperations = std::make_unique<RTensorOperations>(fShuffle, fSetSeed);
      fBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumDatasetCols, fDNumEntries, fDropRemainder);
      
      fNumDataFrames = f_rdfs.size();
      fSumVecSizes = std::accumulate(vecSizes.begin(), vecSizes.end(), 0);
      fNumDatasetCols = fNumColumns + fSumVecSizes - vecSizes.size();

      if (fMixtureWeights.size() > 0) {
         fDataframeBalance = true;
         std::cout << "Weights provided" << std::endl;
      }
      if (fNumDataFrames == 1) {
        fNumEntries = f_rdfs[0].Count().GetValue();
        fEntries = f_rdfs[0].Take<ULong64_t>("rdfentry_");

        // add the last element in entries to not go out of range when filling chunks
        fEntries->push_back((*fEntries)[fNumEntries - 1] + 1);
      
        // if (fBatchSize == 0) {
        //   fBatchSize = fNumEntries;
        // }

        // std::cout << fNumDataFrames << " dataframes" << std::endl; 

        // // number of training and validation entries after the split
        // fNumValidationEntries = static_cast<std::size_t>(fValidationSplit * fNumEntries);
        // fNumTrainingEntries = fNumEntries - fNumValidationEntries;
      
        // fLeftoverTrainingBatchSize = fNumTrainingEntries % fBatchSize;
        // fLeftoverValidationBatchSize = fNumValidationEntries % fBatchSize;

        // fNumFullTrainingBatches = fNumTrainingEntries / fBatchSize;
        // fNumFullValidationBatches = fNumValidationEntries / fBatchSize;

        // fNumLeftoverTrainingBatches = fLeftoverTrainingBatchSize == 0 ? 0 : 1;
        // fNumLeftoverValidationBatches = fLeftoverValidationBatchSize == 0 ? 0 : 1;

        // if (dropRemainder) {
        //   fNumTrainingBatches = fNumFullTrainingBatches;
        //   fNumValidationBatches = fNumFullValidationBatches;
        // }

        // else {
        //   fNumTrainingBatches = fNumFullTrainingBatches + fNumLeftoverTrainingBatches;
        //   fNumValidationBatches = fNumFullValidationBatches + fNumLeftoverValidationBatches;
        // }

      
        if (loadEager) {
           // for (auto rdf : f_rdfs) {
           //    std::size_t NumEntries = rdf.Count().GetValue();
           //    ROOT::RDF::RResultPtr<std::vector<ULong64_t>> Entries = rdf.Take<ULong64_t>("rdfentry_");
           //    // add the last element in entries to not go out of range when filling chunks
           //    Entries->push_back((*Entries)[NumEntries - 1] + 1);
           //    fDataframeEntries.push_back(NumEntries);
           //    // if (fBatchSize == 0) {
           //    //    fBatchSize = fNumEntries;
           //    // }
              
           //    fDatasetLoader =
           //       std::make_unique<RDatasetLoader<Args...>>(rdf, NumEntries, Entries, fValidationSplit,
           //                                                 fCols, vecSizes, vecPadding, fShuffle, fSetSeed);
              
           //    TMVA::Experimental::RTensor<float> TrainDatasetTensor({0, 0});
           //    TMVA::Experimental::RTensor<float> ValidationDatasetTensor({0, 0});   
           //    fDatasetLoader->SplitDataset(TrainDatasetTensor, ValidationDatasetTensor);
              
           //    fTrainingTensors.push_back(TrainDatasetTensor);
           //    fValidationTensors.push_back(ValidationDatasetTensor);
           //    fDataframeTrainingEntries.push_back(TrainDatasetTensor.GetShape()[0]);
           //    fDataframeValidationEntries.push_back(ValidationDatasetTensor.GetShape()[0]);
              
           // }
           // fBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumDatasetCols);
           // fTrainingBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumDatasetCols);
           // fValidationBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumDatasetCols);


          fDatasetLoader =
            std::make_unique<RDatasetLoader<Args...>>(f_rdfs[0], fValidationSplit,
                                                      fCols, vecSizes, vecPadding, fShuffle, fSetSeed);

          
          TMVA::Experimental::RTensor<float> TrainingDataset({0, 0});
          TMVA::Experimental::RTensor<float> ValidationDataset({0, 0});
          fDatasetLoader->SplitDataset(TrainingDataset, ValidationDataset);
          fTrainDatasetTensor = TrainingDataset;
          fValidationDatasetTensor = ValidationDataset;
          
          fTrainingTensors.push_back(fTrainDatasetTensor);
          fValidationTensors.push_back(fValidationDatasetTensor);            

          std::size_t NumTrainingEntries = fTrainDatasetTensor.GetShape()[0];
          std::size_t NumValidationEntries = fValidationDatasetTensor.GetShape()[0];          
          
          fTrainingBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumDatasetCols, NumTrainingEntries, fDropRemainder);
          fValidationBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumDatasetCols, NumValidationEntries, fDropRemainder);
        }

        else {
          fChunkLoader =
            std::make_unique<RChunkLoader<Args...>>(f_rdfs[0], fChunkSize, fBlockSize, fValidationSplit,
                                                    fCols, vecSizes, vecPadding, fShuffle, fSetSeed);
          // split the dataset into training and validation sets
          fChunkLoader->SplitDataset();
          std::size_t NumTrainingEntries = fChunkLoader->GetNumTrainingEntries();
          std::size_t NumValidationEntries = fChunkLoader->GetNumValidationEntries();
          
          fChunkTrainingBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumDatasetCols, NumTrainingEntries, fDropRemainder);
          fChunkValidationBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumDatasetCols, NumValidationEntries, fDropRemainder);


          // number of training and validation chunks, calculated in RChunkConstructor
          fNumTrainingChunks = fChunkLoader->GetNumTrainingChunks();
          fNumValidationChunks = fChunkLoader->GetNumValidationChunks();

          fTrainingChunkNum = 0;
          fValidationChunkNum = 0;
        }
        
      }
      
      
      
      else if (fNumDataFrames > 0) {
      
        std::cout << "It works here" << std::endl;
           
        for (auto rdf : f_rdfs) {
          fNumEntries = rdf.Count().GetValue();
          fEntries = rdf.Take<ULong64_t>("rdfentry_");
          // add the last element in entries to not go out of range when filling chunks
          fEntries->push_back((*fEntries)[fNumEntries - 1] + 1);
          fDataframeEntries.push_back(fNumEntries);
          if (fBatchSize == 0) {
            fBatchSize = fNumEntries;
          }
             
          fDatasetLoader =
            std::make_unique<RDatasetLoader<Args...>>(rdf, fValidationSplit,
                                                      fCols, vecSizes, vecPadding, fShuffle, fSetSeed);
          TMVA::Experimental::RTensor<float> TrainDatasetTensor({0, 0});
          TMVA::Experimental::RTensor<float> ValidationDatasetTensor({0, 0});   

          fDatasetLoader->SplitDataset(TrainDatasetTensor, ValidationDatasetTensor);
          fTrainingTensors.push_back(TrainDatasetTensor);
          fValidationTensors.push_back(ValidationDatasetTensor);            
        }
        std::cout << "RDataFrame entries: "; 
        for (int i = 0; i < fDataframeEntries.size(); i++) {
          std::cout << fDataframeEntries[i] << " ";
        }
        std::cout << std::endl;
      }


   }

   ~RBatchGenerator() { DeActivate(); }

   void DeActivate()
   {
      {
         std::lock_guard<std::mutex> lock(fIsActiveMutex);
         fIsActive = false;
      }

      // fChunkTrainingBatchLoader->DeActivate();
      // fChunkValidationBatchLoader->DeActivate();            
      // fTrainingBatchLoader->DeActivate();
      // fValidationBatchLoader->DeActivate();
      fBatchLoader->DeActivate();

      if (fLoadingThread) {
         if (fLoadingThread->joinable()) {
            fLoadingThread->join();
         }
      }
   }

   /// \brief Activate the loading process by starting the batchloader, and
   /// spawning the loading thread.
   void Activate()
   {
      if (fIsActive)
         return;

      {
         std::lock_guard<std::mutex> lock(fIsActiveMutex);
         fIsActive = true;
      }
      // fChunkTrainingBatchLoader->Activate();      
      // fChunkValidationBatchLoader->Activate();
      // fTrainingBatchLoader->Activate();
      // fValidationBatchLoader->Activate();
      fBatchLoader->Activate();
      // fLoadingThread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunks, this);
   }

   void ActivateEpoch() { fEpochActive = true; }

   void DeActivateEpoch() { fEpochActive = false; }

   void ActivateTrainingEpoch() { fTrainingEpochActive = true; }

   void DeActivateTrainingEpoch() { fTrainingEpochActive = false; }

   void ActivateValidationEpoch() { fValidationEpochActive = true; }

   void DeActivateValidationEpoch() { fValidationEpochActive = false; }

   /// \brief Create training batches by first loading a chunk (see RChunkLoader) and split it into batches (see RBatchLoader)
   void CreateTrainBatches()
   {
     if (fLoadEager) {
      if (fNumDataFrames == 1) {

        std::size_t NumTrainingEntries = fTrainingBatchLoader->GetNumEntries();
        TMVA::Experimental::RTensor<float> ShuffledTrainDatasetTensor({NumTrainingEntries, fNumDatasetCols});
        fRSampler->RandomSampler(ShuffledTrainDatasetTensor, fTrainingTensors);
        // fTensorOperations->ShuffleTensor(ShuffledTrainDatasetTensor, fTrainDatasetTensor);
        fTrainingEpochActive = true;
        std::size_t lastTrainingBatch = 1;
        // fTrainingBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumDatasetCols, fNumEntries, fDropRemainder);         
        // fTrainingBatchLoader->CreateBatches(ShuffledTrainDatasetTensor, 1, fLeftoverTrainingBatchSize, fQueue);
        fTrainingBatchLoader->CreateBatches(ShuffledTrainDatasetTensor, 1, fQueue);
          // fTrainingBatchLoader->FillBatchesInQueue();
      }
      
     }
     else {
       fChunkLoader->CreateTrainingChunksIntervals();
       fTrainingEpochActive = true;
       fTrainingChunkNum = 0;
       fChunkLoader->LoadTrainingChunk(fTrainChunkTensor, fTrainingChunkNum);
       std::size_t lastTrainingBatch = fNumTrainingChunks - fTrainingChunkNum;
       // fChunkTrainingBatchLoader->CreateBatches(fTrainChunkTensor, lastTrainingBatch, fLeftoverTrainingBatchSize, fQueue);
       fChunkTrainingBatchLoader->CreateBatches(fTrainChunkTensor, lastTrainingBatch, fQueue);
       // fChunkTrainingBatchLoader->FillBatchesInQueue();
       fTrainingChunkNum++;

     }
   }


   /// \brief Create training batches by first loading a chunk (see RChunkLoader) and split it into batches (see RBatchLoader)
   void CreateValidationBatches()
   {

      if (fLoadEager) {

         std::size_t NumValidationEntries = fValidationBatchLoader->GetNumEntries();
         TMVA::Experimental::RTensor<float> ShuffledValidationDatasetTensor({NumValidationEntries, fNumDatasetCols});

         fTensorOperations->ShuffleTensor(ShuffledValidationDatasetTensor, fValidationDatasetTensor);         
         fValidationEpochActive = true;
         std::size_t lastValidationBatch = 1;
         // fValidationBatchLoader->CreateBatches(ShuffledValidationDatasetTensor, 1, fLeftoverValidationBatchSize, fQueue);
         fValidationBatchLoader->CreateBatches(ShuffledValidationDatasetTensor, 1, fQueue);
         // fValidationBatchLoader->FillBatchesInQueue();
         auto batchQueue = fValidationBatchLoader->GetNumBatchQueue();
         std::cout << "val set " << ShuffledValidationDatasetTensor.GetShape()[0] << std::endl; 
         std::cout << "validation batchs " << batchQueue << std::endl;          
         // std::cout << "Validation entries (not used) " << fLeftoverValidationBatchSize << std::endl; 
      }

      else {
         fChunkLoader->CreateValidationChunksIntervals();
         fValidationEpochActive = true;
         fValidationChunkNum = 0;
         fChunkLoader->LoadValidationChunk(fValidationChunkTensor, fValidationChunkNum);
         std::size_t lastValidationBatch = fNumValidationChunks - fValidationChunkNum;
         // fChunkValidationBatchLoader->CreateBatches(fValidationChunkTensor, lastValidationBatch, fLeftoverValidationBatchSize, fQueue);
         fChunkValidationBatchLoader->CreateBatches(fValidationChunkTensor, lastValidationBatch, fQueue);
         // fChunkValidationBatchLoader->FillBatchesInQueue();         
         fValidationChunkNum++;

      }
   }
   
   /// \brief Loads a training batch from the queue
   TMVA::Experimental::RTensor<float> GetTrainBatch()
   {
      if (fLoadEager) {
         // Get next batch if available
         if (fNumDataFrames == 1) {
            return fTrainingBatchLoader->GetBatch();            
         }

         else {
            return fSampleTrainingBatchLoader->GetBatch();            
            std::cout << "Is here" << std::endl;
         }

      }

      else {
         auto batchQueue = fChunkTrainingBatchLoader->GetNumBatchQueue();

         // load the next chunk if the queue is empty
         if (batchQueue < 1 && fTrainingChunkNum < fNumTrainingChunks) {
            fChunkLoader->LoadTrainingChunk(fTrainChunkTensor, fTrainingChunkNum);
            std::size_t lastTrainingBatch = fNumTrainingChunks - fTrainingChunkNum;
            // fChunkTrainingBatchLoader->CreateBatches(fTrainChunkTensor, lastTrainingBatch, fLeftoverTrainingBatchSize, fQueue);
            fChunkTrainingBatchLoader->CreateBatches(fTrainChunkTensor, lastTrainingBatch, fQueue);            
            // fChunkTrainingBatchLoader->FillBatchesInQueue();
            fTrainingChunkNum++;
         }

         else {
            std::size_t NumTrainingEntries = fChunkTrainingBatchLoader->GetNumEntries();
            std::size_t NumValidationEntries = fChunkValidationBatchLoader->GetNumEntries();
            std::size_t NumEntries = NumTrainingEntries + NumValidationEntries;
            ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdfs[0], 0, NumEntries);
         }

         // Get next batch if available
         return fChunkTrainingBatchLoader->GetBatch();
      }
   }

   /// \brief Loads a training batch from the queue
   TMVA::Experimental::RTensor<float> GetValidationBatch()
   {
      if (fLoadEager) {
         // Get next batch if available
         return fValidationBatchLoader->GetBatch();
      }

      else {
         auto batchQueue = fChunkValidationBatchLoader->GetNumBatchQueue();

         // load the next chunk if the queue is empty
         if (batchQueue < 1 && fValidationChunkNum < fNumValidationChunks) {
            fChunkLoader->LoadValidationChunk(fValidationChunkTensor, fValidationChunkNum);
            std::size_t lastValidationBatch = fNumValidationChunks - fValidationChunkNum;
            // fChunkValidationBatchLoader->CreateBatches(fValidationChunkTensor, lastValidationBatch, fLeftoverValidationBatchSize, fQueue);
            fChunkValidationBatchLoader->CreateBatches(fValidationChunkTensor, lastValidationBatch, fQueue);
            // fChunkValidationBatchLoader->FillBatchesInQueue();            
            fValidationChunkNum++;
         }

         else {
            std::size_t NumTrainingEntries = fChunkTrainingBatchLoader->GetNumEntries();
            std::size_t NumValidationEntries = fChunkValidationBatchLoader->GetNumEntries();
            std::size_t NumEntries = NumTrainingEntries + NumValidationEntries;
            ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdfs[0], 0, NumEntries);
         }

         // Get next batch if available
         return fChunkValidationBatchLoader->GetBatch();
      }
   }
   
  void distributeBatch() {

    fSumNumEntries = std::accumulate(fDataframeEntries.begin(), fDataframeEntries.end(), 0);
    std::cout << fSumNumEntries << std::endl;

    if (fDataframeBalance) {
       fDataframeBatchFractions = fMixtureWeights;
    }
    
    else {
       for (float i : fDataframeEntries) {
          fDataframeBatchFractions.push_back(i / fSumNumEntries);
       }
    }

    for (float i : fDataframeBatchFractions) {
      std::size_t batchSize = static_cast<std::size_t>(i * fBatchSize);
      fDataframeBatchSizes.push_back(batchSize);
    }
    
    fDataframeBatchSizes.resize(fDataframeBatchFractions.size());
    std::vector<std::pair<float, std::size_t>> remainders; // {fractional part, index}

    // Step 1: multiply by batch size and take floor
    std::size_t current_sum = 0;
    for (std::size_t i = 0; i < fDataframeBatchFractions.size(); i++) {
      float val = fDataframeBatchFractions[i] * fBatchSize;
      fDataframeBatchSizes[i] = static_cast<std::size_t>(std::floor(val));
      remainders.push_back({val - fDataframeBatchSizes[i], i});
      current_sum += fDataframeBatchSizes[i];
    }

    // Step 2: calculate how much we are short
    std::size_t leftover = fBatchSize - current_sum;

    // Step 3: distribute leftover 1s to largest remainders
    std::sort(remainders.rbegin(), remainders.rend()); // descending
    for (std::size_t i = 0; i < leftover; i++) {
      fDataframeBatchSizes[remainders[i].second]++;
    }
  }
  
   
   // std::size_t NumberOfTrainingBatches() { return fNumTrainingBatches; }
   // std::size_t NumberOfValidationBatches() { return fNumValidationBatches; }

   // std::size_t TrainRemainderRows() { return fLeftoverTrainingBatchSize; }
   // std::size_t ValidationRemainderRows() { return fLeftoverValidationBatchSize; }

   std::size_t NumberOfTrainingBatches() {
      if (fLoadEager) {
         return fTrainingBatchLoader->GetNumBatches();
      }         
      else {
         return fChunkTrainingBatchLoader->GetNumBatches();
      }         
   }

   std::size_t NumberOfValidationBatches() {
      if (fLoadEager) {
         return fValidationBatchLoader->GetNumBatches();
      }         
      else {
         return fChunkValidationBatchLoader->GetNumBatches();
      }         
   }
   
   std::size_t TrainRemainderRows() {
      if (fLoadEager) {
         return fTrainingBatchLoader->GetNumRemainderRows();
      }
      else {
         return fChunkTrainingBatchLoader->GetNumRemainderRows();
      }
   }

   std::size_t ValidationRemainderRows() {
      if (fLoadEager) {
         return fValidationBatchLoader->GetNumRemainderRows();
      }
      else {
         return fChunkValidationBatchLoader->GetNumRemainderRows();
      }
   }
   
   bool IsActive() { return fIsActive; }
   bool TrainingIsActive() { return fTrainingEpochActive; }
   /// \brief Returns the next batch of validation data if available.
   /// Returns empty RTensor otherwise.
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_RBATCHGENERATOR
