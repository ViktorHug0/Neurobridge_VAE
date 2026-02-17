    
###########################################################

x_dom = model.backbone(img_dom, track_bn=is_target) # Returns the backbone features as a flat vector

# Encode: backbone features → latent z (split into subspaces) → classifier/domain logits
# Returns: z, z_s1, z_s2, z_c1, z_c2, label_logit, domain_logit, mu, log_var
z, z_s1, z_s2, z_c1, z_c2, label_logit, domain_logit, mu, log_var = model.encode(
    x=x_dom, u=d_dom, track_bn=is_target)

# Store class-related features (z_c1) for later centroid alignment
if is_target:
    tgt_feat = z_c1
else:
    src_feat_list.append(z_c1)

q_dist = torch.distributions.Normal(
    mu, torch.exp(torch.clamp(log_var, min=-10) / 2))
log_qz = q_dist.log_prob(z)
log_pz = normal_distribution.log_prob(z)
kl = (log_qz.sum(dim=1) - log_pz).mean()
C = torch.clamp(torch.tensor(args.C_max) /
                args.C_stop_iter * total_iter, 0, args.C_max)
loss_kl = args.beta * (kl - C).abs()

# VAE RECONSTRUCTION: decode z back to backbone feature space and compute MSE
x_all = torch.cat(x_all, 0)        # Concatenate all backbone features
z_all = torch.cat(z_all, 0)        # Concatenate all latent codes
x_all_hat = model.decode(z_all)    # Reconstruct features from z

# VAE losses
mean_loss_recon = F.mse_loss(
    x_all, x_all_hat, reduction="sum") / len(x_all)  # Reconstruction accuracy
mean_loss_kl = torch.stack(losses_kl, dim=0).mean()  # Average KL across domains
mean_loss_vae = mean_loss_recon + mean_loss_kl       # Total VAE objective

# TOTAL LOSS: weighted combination of all objectives
loss = (mean_loss_cls                              # Supervised source classification
    + args.lambda_dom * mean_loss_domain       # Domain loss (all domains)
    + args.lambda_vae * mean_loss_vae              # VAE (reconstruction + KL)
    + args.lambda_sem * total_semantic_loss  # Centroid alignment (source-target)
    + args.mcc_weight * mcc_loss)                  # Minimum class confusion (target diversity)

########################################################################################