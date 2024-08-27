// Fill out your copyright notice in the Description page of Project Settings.

#include "Generators/Image/GTDepthImageGeneratorComponent.h"

#include <CanvasItem.h>
#include <CanvasTypes.h>
#include <Engine/TextureRenderTarget2D.h>


#include "Generators/Image/GTSceneCaptureComponent2D.h"



UGTDepthImageGeneratorComponent::UGTDepthImageGeneratorComponent()
    : Super()
    , MaxZ(100000.f)
{
    bAntiAliasing = false;
}

void UGTDepthImageGeneratorComponent::DrawDebug(FViewport* Viewport, FCanvas* Canvas)
{
    if (SceneCaptureComponent && SceneCaptureComponent->TextureTarget &&
        SceneCaptureComponent->TextureTarget->IsValidLowLevel())
    {
        UTextureRenderTarget2D* DebugTextureTarget = SceneCaptureComponent->TextureTarget;

        if (DebugTextureTarget)
        {
            FTextureRenderTargetResource* RenderTargetResource =
                DebugTextureTarget->GameThread_GetRenderTargetResource();

            // Ensure the resource is valid before drawing
            if (RenderTargetResource)
            {
                FCanvasTileItem TileItem(
                    FVector2D::ZeroVector,
                    RenderTargetResource,
                    FVector2D(DebugTextureTarget->SizeX, DebugTextureTarget->SizeY),
                    FLinearColor::White);

                TileItem.BlendMode = SE_BLEND_Opaque;
                Canvas->DrawItem(TileItem);
            }
            else
            {
                // Optionally log an error or provide some fallback visualization
                UE_LOG(LogTemp, Warning, TEXT("RenderTargetResource is invalid."));
            }
        }
        else
        {
            // Optionally log an error or provide some fallback visualization
            UE_LOG(LogTemp, Warning, TEXT("DebugTextureTarget is invalid."));
        }
    }
    else
    {
        // Optionally log an error or provide some fallback visualization
        UE_LOG(LogTemp, Warning, TEXT("SceneCaptureComponent or TextureTarget is invalid."));
    }
}





void UGTDepthImageGeneratorComponent::BeginPlay()
{
    Super::BeginPlay();

    SceneCaptureComponent->SetupDepthPostProccess(MaxZ, bUsePerspectiveDepth);
}
