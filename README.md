# Overview

This project implements GPT-style tokenization mechanisms from first principles to understand how transformer models convert raw text into token IDs.
It includes: 
  •Word-level tokenization
  
  •UTF-8 byte-level tokenization
  
  •Byte Pair Encoding (BPE) training
  
  •Merge-statistics based vocabulary construction
  
  •Custom encode/decode pipeline
  
  •Comparison with GPT-2 and GPT-4 tokenizers using tiktoken library

  ## Architecture
  Raw Text → UTF-8 Bytes → BPE Merges → Token IDs → Transformer Input

  ## What This Project Demonstrates
  •Deep understanding of GPT tokenization internals
  
  •Subword tokenization via Byte Pair Encoding

  •Vocabulary construction from raw text

  •Compression analysis

  •Differences between GPT-2 and GPT-4 tokenization

## Run the following commands to run the project
pip install -r requirements.txt

python demo.py
