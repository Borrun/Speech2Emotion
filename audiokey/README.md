
# Speech To Emotion

The purpose of this project is to enable AI to express emotions visually, improving the overall user interaction experience.

**AudioKey** is a lightweight tool for extracting **emotion/expression trigger timestamps (keyframes)** from speech audio.

This repository is only a sub-module of the overall **Speech → Emotion** pipeline. Its goal is to output **when to trigger** an expression, rather than executing the expression itself.

## Status
Work in progress (WIP).

## Data Format
Each reply corresponds to a `.wav` file, and has one label entry in the annotation file:

```json
{"wav":"utt_0001.wav","fps":30,"key_frames":[5,55]}
````

* `fps`: fixed at 30
* `key_frames`: up to 3 keyframes (frame indices) per utterance

(Currently, text and the agent's internal emotion state are not considered.)

## Notes

* Designed for real-world deployment with external TTS APIs using chunked/streaming audio input
* The overall project is still under active development


# Speech To Emotion

本工程目地是让AI具有视觉上的情绪表达，优化交互体验。

**AudioKey** 是一个用于从语音音频中提取 **情绪/表情触发时刻（关键帧）** 的小工具

本仓库只是 **Speech → Emotion** 整体链路中的一个子模块，目标是输出“什么时候触发”，而不是执行表情本身。

## 当前状态
开发中（WIP）。

## 数据格式
每条回复对应一个 `.wav`，并在标注文件中有一行记录：

```json
{"wav":"utt_0001.wav","fps":30,"key_frames":[5,55]}
````

* `fps`：固定为 30
* `key_frames`：每条最多 3 个关键帧（帧号）

（目前并未考虑文本与agnet的当前情绪状态）

## 备注

* 面向API调外部TTS的chunk/流式输入的落地场景设计
* 整体工程仍在持续完善中
