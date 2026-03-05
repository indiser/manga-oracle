# Quick Reference: Manga Survival Prediction Engine

## 🚀 Three-Command Quick Start

```bash
# 1. Navigate to the folder
cd "C:\Users\ranab\OneDrive\Desktop\Manga Surrvival"

# 2. Run the engine
python survival_engine.py

# 3. Check results in:
#    - manga_analysis_reports/     (charts & visualizations)
#    - manga_predictions/           (individual predictions)
#    - SUMMARY_REPORT.txt           (executive summary)
```

---

## 📊 What Each Output File Means

| File | What It Shows | How to Use It |
|------|---------------|--------------| 
| `roc_all_models.png` | Model performance comparison | Higher curves = better models |
| `calibration_curves.png` | Are predictions reliable? | Closer to diagonal = trustworthy |
| `feature_importance_top20.png` | What drives predictions? | Tall bars = important factors |
| `confusion_matrices.png` | True positives/negatives | Understand prediction types |
| `model_comparison_table.png` | Quick stats on all models | Compare metrics side-by-side |
| `survival_global_curve.png` | How long manga last | Understand industry trends |

---

## 🎯 Risk Levels (Plain English)

### 🟢 VERY LOW RISK (0-10%)
- Almost certainly will continue
- Keep doing what you're doing
- Focus on maintaining quality

### 🟢 LOW RISK (10-25%)
- Likely to continue for a long time
- Good trajectory
- Standard maintenance

### 🟡 MODERATE RISK (25-50%)
- Could go either way
- Watch the metrics
- Make improvements if possible

### 🔴 HIGH RISK (50-75%)
- Likely to be cancelled
- Take action now
- Review feedback and adjust

### 🔴 VERY HIGH RISK (75-100%)
- Almost certainly will be cancelled
- Emergency measures needed
- Major changes required

---

## 💡 What "AUC = 0.82" Means

**Translation:** 
- If we randomly pick a cancelled manga and a continuing manga
- The model correctly identifies which is which 82% of the time
- This is "very good" for complex predictions

**In context:**
- 0.5 = Coin flip (useless)
- 0.7 = Decent
- 0.8 = Very good ⭐
- 0.9 = Excellent
- 1.0 = Perfect (impossible)

---

## 🔍 Feature Importance Explained

When you see:
```
Score ████████░░░░ 25%
Members ██████░░░░░░ 15%
Duration █████░░░░░░░ 12%
```

It means:
- **Score (Quality)** has 25% influence on prediction
- **Members (Popularity)** has 15% influence
- **Duration (How long it ran)** has 12% influence

**Key insight:** Quality is the strongest predictor of survival.

---

## 🎓 6 Models Explained Simply

| # | Model | Think Of It As | Strength |
|---|-------|---|---|
| 1 | **Logistic Regression** | A straight line through data | Simple, fast, interpretable |
| 2 | **Random Forest** | 100 decision trees voting | Handles complexity well |
| 3 | **Gradient Boosting** | Trees learning from mistakes | Captures patterns excellently |
| 4 | **XGBoost** | Optimized version of #3 | State-of-the-art performance |
| 5 | **Voting Ensemble** | All 4 models voting together | Stable, democratic |
| 6 | **Stacking Ensemble** | Smart combination of 1-3 | Best overall usually |

**The engine automatically picks the best one for your data.**

---

## 📝 Reading a Prediction

```
╔════════════════════════════════════════════════════╗
║  PREDICTION: My Favorite Manga                    ║
╚════════════════════════════════════════════════════╝

🟡 MODERATE RISK (42% cancellation chance)

↓ What to do:
  ✓ Increase marketing
  ✓ Improve chapter quality
  ✓ Engage with readers

↓ Why this prediction:
  • Popular (ranks in top 25%)
  • Good quality (7.2/10 score)
  • Medium duration (running 3+ years)
```

---

## 🛠️ Troubleshooting

### Engine won't run
**Error:** `ModuleNotFoundError: No module named 'pandas'`  
**Fix:** 
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lifelines
```

### Error about missing CSV
**Error:** `File 'final_manga_dataset_clean.csv' not found`  
**Fix:** Make sure the CSV is in the same folder as `survival_engine.py`

### Too slow or out of memory
**Fix:**
1. Set `Config.USE_XGBOOST = False` 
2. Set `Config.CV_FOLDS = 3` (instead of 5)
3. Reduce dataset size

---

## 📈 Key Metrics Summary

| Metric | Range | Interpretation |
|--------|-------|---|
| **AUC** | 0.0-1.0 | How well model separates classes (0.8 = excellent) |
| **Accuracy** | 0%-100% | % of correct predictions |
| **Precision** | 0%-100% | Of predicted cancellations, % that were right |
| **Recall** | 0%-100% | Of actual cancellations, % we found |
| **F1 Score** | 0%-100% | Balance between precision & recall |
| **Brier Score** | 0-1 | Probability prediction error (lower = better) |

---

## 🔮 What Predictions CAN Do

✅ Identify at-risk manga early  
✅ Understand survival factors  
✅ Compare multiple manga  
✅ Guide strategic decisions  
✅ Provide data-backed insights  

## What Predictions CAN'T Do

❌ Predict with 100% certainty  
❌ Account for author illness/death  
❌ Predict viral popularity  
❌ Factor in behind-the-scenes politics  
❌ Guarantee outcomes  

**Use as ONE data point, not the final say.**

---

## 📊 50+ Features Created

The engine automatically creates (examples):

**Time-based:** Duration, start year, launch month, season  
**Text-based:** Title length, word count, complexity  
**Quality:** Score, score squared, log transformations  
**Popularity:** Members, log members, z-score  
**Genre:** Tag count, demographic indicators  
**Interactions:** Score × Members, Demographic × Quality  
**Polynomials:** Higher-order effects  

**Total: 50+ features from raw data**

---

## 🎯 Action Items by Risk Level

### 🟢 Low Risk
- [ ] Maintain current quality
- [ ] Track metrics monthly
- [ ] Plan long-term content roadmap

### 🟡 Moderate Risk
- [ ] Analyze reader feedback
- [ ] Test new content ideas
- [ ] Boost marketing efforts
- [ ] Monitor metrics weekly

### 🔴 High Risk
- [ ] Emergency meeting with team
- [ ] Review all metrics in detail
- [ ] Plan major changes
- [ ] Increase engagement
- [ ] Consider format adjustments

### 🔴 Very High Risk
- [ ] Immediate action required
- [ ] Major storyline changes
- [ ] Aggressive marketing campaign
- [ ] Secure editorial support
- [ ] Daily monitoring

---

## 💾 Saving & Using Trained Models

Models are automatically saved to `manga_models/` folder:

```python
# To use a saved model later:
import pickle

# Load the model
with open('manga_models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load feature names (important!)
with open('manga_models/feature_names.pkl', 'rb') as f:
    features = pickle.load(f)

# Load scaler (for consistent preprocessing)
with open('manga_models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Now use model.predict(new_data) on similar manga
```

---

## 📞 When to Retrain

**Every Month:**
- New manga data arrives
- Industry trends shift
- Keep predictions current

**Every Quarter:**
- Major market shifts detected
- Prediction accuracy drops
- New evaluation approaches

**Every Year:**
- Full pipeline refresh
- Consider new features
- Update algorithms

---

## 🏆 Best Practices

1. **Trust But Verify**
   - Use predictions + human judgment
   - Check qualitative factors too
   - Monitor actual outcomes

2. **Regular Monitoring**
   - Check predictions monthly
   - Update as new data arrives
   - Note prediction accuracy

3. **Context Matters**
   - 45% risk might be low for action manga
   - But high for psychological manga
   - Consider genre norms

4. **Communicate Results**
   - Show stakeholders the uncertainty
   - Explain what "60% risk" means
   - Use plain English explanations

---

## 📚 Further Reading

In this folder:
- `SURVIVAL_ENGINE_GUIDE.md` - Full technical guide
- `survival_engine.py` - Annotated source code
- Output files explain themselves

---

**Remember:** This is a tool to support decisions, not replace judgment. 🎯
