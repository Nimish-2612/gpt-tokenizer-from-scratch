GPT Tokenizer From Scratch
Overview

This project implements GPT-style tokenization mechanisms from scratch, including:

Word-level tokenization

UTF-8 byte tokenization

Byte Pair Encoding (BPE)

Merge statistics

Custom encode/decode pipeline

Comparison with OpenAI’s GPT-2 and GPT-4 tokenizers

What This Project Demonstrates

Deep understanding of how transformers process text

Vocabulary construction

Token compression via BPE

Differences between GPT-2 and GPT-4 tokenization

Regex-based forced token splits

Practical compression ratio analysis

Architecture

Text → UTF-8 Bytes → BPE Merges → Token IDs → Transformer Input

Results Example
Method	Tokens
Raw bytes	240
Custom BPE	132
GPT-2	128
Concepts Covered

Tokenization

Vocabulary construction

Subword tokenization

Compression algorithms

UTF-8 encoding

Regex token splitting

GPT tokenization differences