# Will use this to develop a mixed effects model to compare backbones and losses betweeen SegFormer instances
# Goal is to show the difference in Mean, Sediment, and Whitewater IoU between the 10 loss functions, and the two backbone sizes

set.seed(42)

library(ARTool)
library(tidyverse)

mean_iou <- "/Users/clayc/Documents/Work/HazShoalWaves/TrainDatasets/images_and_labels_RGB/trained_models/mean_iou_per_scene.csv"
sediment_iou <- "/Users/clayc/Documents/Work/HazShoalWaves/TrainDatasets/images_and_labels_RGB/trained_models/sediment_iou_per_scene.csv"
whitewater_iou <- "/Users/clayc/Documents/Work/HazShoalWaves/TrainDatasets/images_and_labels_RGB/trained_models/whitewater_iou_per_scene.csv"

# these should lead to a factorial study of 2x5x10 with ART ANOVA
mean_iou_df <- read_csv(mean_iou, show_col_types = FALSE)
# sediment_iou_df <- read_csv(sediment_iou, show_col_types = FALSE)
# whitewater_iou_df <- read_csv(whitewater_iou, show_col_types = FALSE)

mean_iou_df$loss <- factor(mean_iou_df$loss, levels = c(unique(mean_iou_df$loss)))
mean_iou_df$backbone <- factor(mean_iou_df$backbone, levels = c(unique(mean_iou_df$backbone)))
mean_iou_df$Image_ID <- factor(mean_iou_df$Image_ID, levels = c(unique(mean_iou_df$Image_ID)))

# mean_run <- interaction(mean_iou_df$loss, mean_iou_df$backbone)
# mean_cfg <- interaction(mean_iou_df$loss, mean_iou_df$backbone, mean_iou_df$seed)

# (1 | Image_ID/mean_run) + (1 | mean_cfg)
mean_model <- art(MeanIoU ~ loss*backbone+(1 | seed), data=mean_iou_df)

summary(mean_model)

# mean_model, sediment_model, whitewater_model
tgt_model <- whitewater_model

# TODO: maybe look at comparing results from m and Wilcoxon tests to show the outcomes are the same?
sim <- simulateResiduals(tgt_model, n=100, group=whitewater_iou_df$Image_ID)
sim_res <- residuals(sim)
plot(sim)
testZeroInflation(sim)

ks.test(sim_res, "punif")$statistic
mean(sim_res < 0.01); mean(sim_res > 0.99)
# plot(sim_res, quantreg = T, form=cfg)

# g <- as.data.frame(emm)
# # g1 <- subset(g, sensor == levels(g$sensor)[1])  # for future use when | sensor is added
# ref_idx <- which(g$loss == "Dice" & g$backbone == "b0")

emm <- regrid(emmeans(tgt_model, ~ loss | backbone, component = "response"))
ct <- contrast(emm, method = "trt.vs.ctrl", ref = "Dice", adjust = "mvt")
summary(ct, infer = TRUE)

emm_bb <- regrid(emmeans(tgt_model, ~ backbone | loss, component = "response"))
contrast(emm_bb, method = "trt.vs.ctrl", ref = "b0", adjust = "mvt")

contrast(emm, interaction = c(loss = "trt.vs.ctrl", backbone = "trt.vs.ctrl"),
         ref = "Dice")

ci <- as.data.frame(summary(ct, infer = TRUE))
names(ci)[grepl("LCL", names(ci))] <- "lcl"
names(ci)[grepl("UCL", names(ci))] <- "ucl"

ggplot(ci, aes(x = estimate, y = reorder(contrast, estimate))) +
  annotate("rect", xmin = -0.03, xmax = 0.03, ymin = -Inf, ymax = Inf, alpha = 0.15) +  # your "too small to matter" band
  geom_vline(xintercept = 0, linetype = "dashed") +
  geom_pointrange(aes(xmin = lcl, xmax = ucl)) +
  facet_wrap(~ backbone) +
  labs(x = "Difference in IoU vs Dice (same backbone)", y = NULL)