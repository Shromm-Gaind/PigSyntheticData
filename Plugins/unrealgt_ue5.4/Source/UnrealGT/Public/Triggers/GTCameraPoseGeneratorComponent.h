#pragma once

#include "CoreMinimal.h"

#include "Generators/GTDataGeneratorComponent.h"

#include "GTCameraPoseGeneratorComponent.generated.h"

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class UNREALGT_API UGTCameraPoseGeneratorComponent : public UGTDataGeneratorComponent
{
    GENERATED_BODY()

public:
    UGTCameraPoseGeneratorComponent();
    // Editable properties for the target's location and rotation
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Target")
    FVector TargetLocation;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Target")
    FRotator TargetRotation;

    virtual void GenerateData(const FDateTime& TimeStamp) override;

private:
    FString Vector3DToFormattedString(const FVector& InVector);
    FString RotatorToFormattedString(const FRotator& InRotator);
    FString MatrixToFormattedString(const FMatrix& InMatrix);
    FString PermutationMatrixToFormattedString(const FMatrix& InMatrix);
};

