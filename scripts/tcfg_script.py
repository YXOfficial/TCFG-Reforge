import torch
import gradio as gr
from modules import scripts

# --- Bắt đầu phần Logic cốt lõi của TCFG ---

# Import ModelPatcher một cách an toàn từ ReForge
try:
    from ldm_patched.modules.model_patcher import ModelPatcher
except ImportError:
    try:
        from backend.patcher.base import ModelPatcher
    except ImportError:
        class ModelPatcher: pass

def score_tangential_damping(cond_score: torch.Tensor, uncond_score: torch.Tensor) -> torch.Tensor:
    """Tính toán score của uncond đã được "damped" bằng cách chiếu nó lên cond_score."""
    batch_num = cond_score.shape[0]
    cond_score_flat = cond_score.reshape(batch_num, 1, -1).float()
    uncond_score_flat = uncond_score.reshape(batch_num, 1, -1).float()

    score_matrix = torch.cat((uncond_score_flat, cond_score_flat), dim=1)
    try:
        _, _, Vh = torch.linalg.svd(score_matrix, full_matrices=False)
    except RuntimeError:
        _, _, Vh = torch.linalg.svd(score_matrix.cpu(), full_matrices=False)

    v1 = Vh[:, 0:1, :].to(uncond_score_flat.device)
    uncond_score_td = (uncond_score_flat @ v1.transpose(-2, -1)) * v1
    return uncond_score_td.reshape_as(uncond_score).to(uncond_score.dtype)

# --- Kết thúc phần Logic cốt lõi của TCFG ---


# --- Bắt đầu Lớp Script cho WebUI ---

class TCFGScript(scripts.Script):
    def title(self):
        """Tiêu đề của script trong giao diện."""
        return "Tangential Damping CFG (TCFG)"

    def show(self, is_img2img):
        """Script này sẽ hiển thị ở cả hai tab txt2img và img2img."""
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        """Tạo giao diện người dùng cho script."""
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label="Enabled", value=False)
            scale = gr.Slider(
                label="TCFG Scale",
                minimum=0.0,
                maximum=2.0, # Cho phép scale lớn hơn 1 một chút để thử nghiệm
                step=0.01,
                value=1.0,
                info="Độ mạnh của việc 'làm thẳng hàng' uncond. 1.0 là theo đúng paper. 0.0 là tắt."
            )
        
        self.infotext_fields = [
            (enabled, "tcfg_enabled"),
            (scale, "tcfg_scale"),
        ]
        
        return [enabled, scale]

    def patch_model(self, model_patcher, scale, cfg_scale):
        """Hàm để patch model, sử dụng hook 'post_cfg'."""
        m = model_patcher.clone()

        def post_cfg_tcfg(args):
            # Lấy các tham số cần thiết từ hook
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            x = args["input"]
            
            # Tính toán score (epsilon) từ prediction
            cond_score = x - cond_pred
            uncond_score = x - uncond_pred
            
            # Tính toán unconditional score đã được "damped"
            uncond_score_td = score_tangential_damping(cond_score, uncond_score)

            # Nội suy giữa score gốc và score đã damped bằng TCFG scale
            final_uncond_score = torch.lerp(uncond_score, uncond_score_td, scale)

            # Tính lại prediction từ score đã được sửa đổi
            final_uncond_pred = x - final_uncond_score

            # Đây là bước quan trọng:
            # Tính toán lại kết quả CFG cuối cùng bằng công thức CFG,
            # nhưng với uncond_pred đã được sửa đổi.
            # `cfg_scale` được lấy từ processing object `p`.
            cfg_result = final_uncond_pred + cfg_scale * (cond_pred - final_uncond_pred)
            
            return cfg_result

        # Đặt hàm hook SAU CFG (post_cfg)
        m.set_model_sampler_post_cfg_function(post_cfg_tcfg)
        return m

    def process_before_every_sampling(self, p, enabled, scale, **kwargs):
        """Hàm này được gọi ngay trước khi quá trình sampling bắt đầu."""
        if not enabled or scale == 0.0:
            return

        try:
            # Lấy đối tượng UNet từ ReForge
            unet = p.sd_model.forge_objects.unet

            # Patch UNet bằng logic TCFG của chúng ta
            # Chúng ta cần cfg_scale từ `p` để tính toán lại chính xác
            patched_unet = self.patch_model(unet, scale, p.cfg_scale)

            # Thay thế UNet gốc bằng UNet đã patch cho lần tạo ảnh này
            p.sd_model.forge_objects.unet = patched_unet

            # Thêm thông số vào ảnh để có thể tái tạo
            p.extra_generation_params.update({
                "tcfg_enabled": enabled,
                "tcfg_scale": scale,
            })
        except Exception as e:
            print(f"[TCFG Script] Error during patching: {e}")
            
        return
