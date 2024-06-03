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

    virtual void GenerateData(const FDateTime& TimeStamp) override;
};


/*
#include "Components/SkeletalMeshComponent.h"  // Make sure to include this
#include "CoreMinimal.h"

#include "Generators/GTDataGeneratorComponent.h"

#include "GTCameraPoseGeneratorComponent.generated.h"

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))

class UNREALGT_API UGTCameraPoseGeneratorComponent : public UGTDataGeneratorComponent
{
    GENERATED_BODY()

public:
    UGTCameraPoseGeneratorComponent();

    virtual void GenerateData(const FDateTime& TimeStamp) override;
};


*/

